import functools
from typing import Literal

import math
import numpy

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from torch.onnx.symbolic_opset11 import unsqueeze

from flcore.trainmodel.lora_modules.lora_layers import LoRALayer


class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate, act="softmax"):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(in_channels, 2 * in_channels)
        self.fc2 = nn.Linear(2 * in_channels, num_experts)
        if act == "softmax":
            self.act = nn.Softmax(dim=-1)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError

    def forward(self, x):
        # x = torch.flatten(x)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.act(x)


class RouteConv2D(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, name, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', client_num=4, dropout_rate=0.2):
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.kernel_size = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.dropout_rate = dropout_rate
        super(RouteConv2D, self).__init__(
            in_channels, out_channels, self.kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self.use_bias = bias
        self.client_num = client_num
        self.num_experts = self.client_num + 1
        self.name = name
        self.reset_parameters()

    def _mix_trajectories(self, state_dicts):
        '''
        paths : [local_models_path1, ..., local_models_pathn, global model]  len: n+1
        '''
        self.reset_parameters()
        self.init_data = []
        self.init_data_bias = []
        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(self.in_channels, self.num_experts, self.dropout_rate)
        self.weight = Parameter(torch.Tensor(
            self.num_experts, self.out_channels, self.in_channels // self.groups, *self.kernel_size))

        for idx in range(self.client_num):
            local_data_weight = state_dicts[idx][self.name + '.weight']
            self.init_data.append(local_data_weight)
            if self.use_bias:
                local_data_bias = state_dicts[idx][self.name + '.bias']
                self.init_data_bias.append(local_data_bias)

        self.init_data.append(state_dicts[-1][self.name + '.weight'])
        self.init_data = torch.stack(self.init_data, dim=0)
        self.weight.data = self.init_data
        if self.use_bias:
            self.init_data_bias.append(state_dicts[-1][self.name + '.bias'])
            self.init_data_bias = torch.stack(self.init_data_bias, dim=0)
            self.bias.data = self.init_data_bias

    def _conv_forward(self, input, weight, bias=None):
        if bias is None:
            bias = self.bias
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, inputs):
        b, _, _, _ = inputs.size()
        res = []
        for inpt in inputs:
            inpt = inpt.unsqueeze(0)
            pooled_inputs = self._avg_pooling(inpt)
            routing_weights = self._routing_fn(pooled_inputs)
            kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)
            biases = torch.sum(routing_weights[:, None] * self.bias, 0) if self.use_bias else None
            out = self._conv_forward(inpt, kernels, biases)
            res.append(out)
        return torch.cat(res, dim=0)


class RouteLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, name: str, bias: bool = True,
                 dropout_rate=0.2) -> None:
        super(RouteLinear, self).__init__(in_features, out_features, bias)
        self.name = name
        self.use_bias = bias
        # self.client_num = client_num
        # self.num_experts = self.client_num + 1
        self.num_experts = None
        self.client_num = None
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate

        self.reset_parameters()

    def _mix_trajectories(self, state_dicts, client_names ):
        '''
        paths : [local_models_path1, ..., local_models_pathn, global model]  len: n+1
        '''
        self.num_experts = len(state_dicts) - 1 # 这里删除掉global的!
        self.init_data = []
        self.init_data_bias = []
        self._avg_pooling = functools.partial(F.adaptive_avg_pool1d, output_size=1)
        self._routing_fn = _routing(self.in_features, self.num_experts, self.dropout_rate)

        self.weight = Parameter(torch.Tensor(
            self.num_experts, self.out_features, self.in_features))

        # for idx in range(self.client_num):
        for name in client_names:
            #
            local_data_weight = state_dicts[name][self.name + '.weight']
            self.init_data.append(local_data_weight)
            if self.use_bias:
                local_data_bias = state_dicts[name][self.name + '.bias']
                self.init_data_bias.append(local_data_bias)

        self.init_data.append(state_dicts['global'][self.name + '.weight'])
        self.init_data = torch.stack(self.init_data, dim=0)
        self.weight.data = self.init_data
        if self.use_bias:
            self.init_data_bias.append(state_dicts['global'][self.name + '.bias'])
            self.init_data_bias = torch.stack(self.init_data_bias, dim=0)
            self.bias.data = self.init_data_bias

    def forward(self, inputs):
        b, _ = inputs.size()
        res = []
        for inpt in inputs:
            inpt = inpt.unsqueeze(0)
            routing_weights = self._routing_fn(inpt)
            kernels = torch.sum(routing_weights[:, None, None] * self.weight, 0)
            biases = torch.sum(routing_weights[:, None] * self.bias, 0) if self.use_bias else None
            out = F.linear(inpt, kernels, biases)
            res.append(out)
        return torch.cat(res, dim=0)


class RouteConvTranspose2D(_ConvTransposeNd):

    def __init__(self, in_channels, out_channels, kernel_size, name, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', client_num=5, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(RouteConvTranspose2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, _pair(0), groups, bias, padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = kernel_size
        self.client_num = client_num
        self.num_experts = self.client_num + 1
        self.dropout_rate = dropout_rate
        self.name = name
        self.use_bias = bias
        self.reset_parameters()

    def _mix_trajectories(self, state_dicts):
        '''
        paths : [local_models_path1, ..., local_models_pathn, global model]  len: n+1
        '''
        self.reset_parameters()
        self.init_data = []
        self.init_data_bias = []
        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(self.in_channels, self.num_experts, self.dropout_rate)
        self.weight = Parameter(torch.Tensor(
            self.num_experts, self.in_channels, self.out_channels // self.groups, *self.kernel_size))
        self.bias = Parameter(torch.Tensor(self.num_experts, self.out_channels))

        for idx in range(self.client_num):
            local_data_weight = state_dicts[idx][self.name + '.weight']
            self.init_data.append(local_data_weight)
            if self.use_bias:
                local_data_bias = state_dicts[idx][self.name + '.bias']
                self.init_data_bias.append(local_data_bias)

        self.init_data.append(state_dicts[-1][self.name + '.weight'])
        self.init_data = torch.stack(self.init_data, dim=0)
        self.weight.data = self.init_data
        if self.use_bias:
            self.init_data_bias.append(state_dicts[-1][self.name + '.bias'])
            self.init_data_bias = torch.stack(self.init_data_bias, dim=0)
            self.bias.data = self.init_data_bias

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return None
        return F.conv_transpose2d(input, weight, bias, self.stride,
                                  self.padding, 0, self.groups, self.dilation)

    def forward(self, inputs):
        b, _, _, _ = inputs.size()
        res = []
        for input in inputs:
            input = input.unsqueeze(0)
            pooled_inputs = self._avg_pooling(input)
            routing_weights = self._routing_fn(pooled_inputs)
            kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)
            biases = torch.sum(routing_weights[:, None] * self.bias, 0)
            out = self._conv_forward(input, kernels, biases)
            res.append(out)
        return torch.cat(res, dim=0)


class ProtoGuidedLoRA(nn.Linear, LoRALayer):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 # num_source_clients: int,
                 name,
                 r: int = 0,
                 lora_alpha: int = 1,
                 lora_dropout: float = 0.,
                 fan_in_fan_out: bool = False,
                 # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
                 merge_weights: bool = True,
                 dropout_rate=0.2,
                 **kwargs):
        """
        参数:
            num_prototypes: 对应预训练阶段的已知癌种数量 (K)
            use_basis_mixing: True则学习基矩阵混合(更强), False则学习对角缩放(更轻)
        """
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        # self.num_source_clients = num_source_clients
        self.fan_in_fan_out = fan_in_fan_out
        self.scaling = lora_alpha / self.r
        self.name = name
        self.dropout_rate = dropout_rate

        # # --- 1. LoRA A (Down-projection) ---
        # self.lora_A = nn.Parameter(self.weight.new_zeros((num_source_clients + 1,r, in_features)))#nn.Linear(in_features, self.r, bias=False)
        # # --- 4. LoRA B (Up-projection) ---
        # self.lora_B = nn.Parameter(self.weight.new_zeros((num_source_clients + 1, out_features, r)))  # 通常输出维度等于输入维度
        assert not fan_in_fan_out

        self.reset_parameters()
        self.enable_lora = True
        self.disable_scaling = False
        self.average_route = False
        self.save_route_scores = False

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)


    def _mix_trajectories(self, state_dicts, frozen_experts=True,
                          route_act: Literal["softmax", "sigmoid"] = 'softmax',
                          disable_scaling: bool = False,
                          average_route: bool = False):
        '''
        paths : [local_models_path1, ..., local_models_pathn, global model]  len: n+1
        '''
        self.disable_scaling = disable_scaling
        self.num_experts = len(state_dicts)
        self.average_route = average_route
        init_data_a = []
        init_data_b = []
        self._avg_pooling = functools.partial(F.adaptive_avg_pool1d, output_size=1)
        if average_route:
            # 直接输出的平均的moe分数
            self._routing_fn_a = lambda in_feature: torch.empty((in_feature.size(0), self.num_experts), device=in_feature.device).fill_(1 / self.num_experts)
            self._routing_fn_b = lambda in_feature: torch.empty((in_feature.size(0), self.num_experts), device=in_feature.device).fill_(1 / self.num_experts)
        else:
            self._routing_fn_a = _routing(self.in_features, self.num_experts, self.dropout_rate, act=route_act)
            self._routing_fn_b = _routing(self.in_features, self.num_experts, self.dropout_rate, act=route_act)

        # self.weight = nn.Parameter(torch.Tensor(
        #     self.num_experts, self.out_features, self.in_features))
        self.lora_A_weight = nn.Parameter(self.weight.new_zeros((self.num_experts, self.r, self.in_features)), requires_grad=not frozen_experts)
        self.lora_B_weight = nn.Parameter(self.weight.new_zeros((self.num_experts, self.out_features, self.r)), requires_grad=not frozen_experts)

        # lora_a = self.weight.new_zeros((self.r, self.in_features))
        # lora_b = self.weight.new_zeros((self.out_features, self.r))
        # nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
        # nn.init.zeros_(lora_b)

        # self.init_data_a.append(lora_a)
        # self.init_data_b.append(lora_b)
        for i, state_dict in enumerate(state_dicts):
            #
            init_data_a.append(state_dict[self.name + '.lora_A'])
            init_data_b.append(state_dict[self.name + '.lora_B'])

        init_data_a = torch.stack(init_data_a, dim=0)
        init_data_b = torch.stack(init_data_b, dim=0)

        self.lora_A_weight.data.copy_(init_data_a)
        self.lora_B_weight.data.copy_(init_data_b)



    def set_lora_state(self, mode):
        self.enable_lora = mode

    def forward(self, x):
        # x: [B, M, D]

        result = F.linear(x, self.weight, self.bias)  # [B,M,D]
        scaling = self.scaling if not self.disable_scaling else 1.0
        if self.enable_lora:
            num_instance_first = False
            h = self.lora_dropout(x)
            if result.ndim == 3:
                num_instance_first = True
                assert h.size(0) == 1
                s = h.squeeze(0)  # [M, in_feature]
            else:
                s = h  # [bs, in_feature]
            bsz = s.size(0)
            routing_weights_a = self._routing_fn_a(s)  # [M, n_exp]
            routing_weights_b = self._routing_fn_b(s)  # [M, n_exp]
            # routing_weights_b = routing_weights_a
            if self.save_route_scores:
                self.saved_route_A = routing_weights_a.detach().cpu()
                self.saved_route_B = routing_weights_b.detach().cpu()
            if num_instance_first:

                # [m, n_feat]
                # lora_a = torch.sum(routing_weights_a[:, :, None, None] * self.lora_A_weight.unsqueeze(0),   #lora_A_weight,unsqueeze -> [1, n_exp, dim1, dim2]
                #                    1)  # routing_weights_a[:, :, None, None]: [M, n_exp, None, None] => [M, n_exp, r, in_feature] -> [M, r, in_feature]
                # lora_b = torch.sum(routing_weights_b[:, :, None, None] * self.lora_B_weight.unsqueeze(0),
                #                    1)  # [M, n_exp,out_dim,r] -> [M, out_dim, r]
                a_shape = (bsz, ) + tuple(self.lora_A_weight.shape[1:])
                b_shape = (bsz, ) +tuple(self.lora_B_weight.shape[1:])
                lora_a = (routing_weights_a @ self.lora_A_weight.view(self.lora_A_weight.size(0), -1)).view(*a_shape) # routing_weights_a[:, :, None, None]: [M, n_exp, None, None] => [M, n_exp, r, in_feature] -> [M, r, in_feature]
                lora_b = (routing_weights_b @ self.lora_B_weight.view(self.lora_B_weight.size(0), -1)).view(*b_shape)


                slices_down = torch.einsum('bmd, bmrd -> bmr', h, lora_a.unsqueeze(0))  # [B,M,D] @ [B=1,M,rank,d] -> [M,1,R]
                slices_up = torch.einsum('bmr, bmor -> bmo', slices_down, lora_b.unsqueeze(0)) * scaling
                return slices_up + result
            else:
                # 2D [bs,n_feat]
                # lora_a = torch.sum(routing_weights_a[0][:, None, None] * self.lora_A_weight, 0)
                # lora_b = torch.sum(routing_weights_b[0][:, None, None] * self.lora_B_weight, 0)
                a_shape = (bsz,) + tuple(self.lora_A_weight.shape[1:])
                b_shape = (bsz,) + tuple(self.lora_B_weight.shape[1:])
                lora_a = (routing_weights_a @ self.lora_A_weight.view(self.lora_A_weight.size(0), -1)).view(
                    *a_shape)  # routing_weights_a[:, :, None, None]: [M, n_exp, None, None] => [M, n_exp, r, in_feature] -> [M, r, in_feature]
                lora_b = (routing_weights_b @ self.lora_B_weight.view(self.lora_B_weight.size(0), -1)).view(*b_shape)
                slices_down = torch.einsum('bd, brd -> br', h, lora_a)  # [B,D] @ [B=1,rank,d]
                slices_up = torch.einsum('br, bor -> bo', slices_down, lora_b) * scaling
                return slices_up + result

        return result

    def extra_repr(self) -> str:
        return (super().extra_repr()) + "," + f"rank={self.r},alpha={self.lora_alpha},num_experts={self.num_experts},disable_scaling={self.disable_scaling},enable_lora={self.enable_lora}"

class MoEConvNDLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, name, r=0, lora_alpha=1, lora_dropout=0.,
                 merge_weights=True, router_drop_out=0.2, **kwargs):
        super(MoEConvNDLoRA, self).__init__()
        self.name = name
        self.router_drop_out = router_drop_out
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            # self.lora_A = nn.Parameter(
            #     self.conv.weight.new_zeros((int(r * out_channels) * kernel_size, in_channels * kernel_size))
            # )
            # self.lora_B = nn.Parameter(
            #     self.conv.weight.new_zeros(
            #         (out_channels // self.conv.groups * kernel_size, int(r * out_channels) * kernel_size))
            # )
            self.scaling = self.lora_alpha / int(self.r * out_channels)
            # Freezing the pre-trained weight matrix
            # self.conv.weight.requires_grad = False
            self.conv.weight.requires_grad = True
        self.reset_parameters()
        self.merged = False
        self.enable_lora = True
        self.disable_scaling = False
        self.average_route = False

    def set_lora_state(self, mode):
        self.enable_lora = mode

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def _mix_trajectories(self, state_dicts, frozen_experts=True,
                          route_act: Literal["softmax", "sigmoid"] = 'softmax',
                          disable_scaling: bool = False,
                          average_route=False):
        '''
        paths : [local_models_path1, ..., local_models_pathn, global model]  len: n+1
        '''
        # self.reset_parameters()
        # self.init_data = []
        # self.init_data_bias = []
        # self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        # self._routing_fn = _routing(self.in_channels, self.num_experts, self.dropout_rate)
        # self.weight = Parameter(torch.Tensor(
        #     self.num_experts, self.out_channels, self.in_channels // self.groups, *self.kernel_size))
        #
        # for idx in range(self.client_num):
        #     local_data_weight = state_dicts[idx][self.name + '.weight']
        #     self.init_data.append(local_data_weight)
        #     if self.use_bias:
        #         local_data_bias = state_dicts[idx][self.name + '.bias']
        #         self.init_data_bias.append(local_data_bias)
        #
        # self.init_data.append(state_dicts[-1][self.name + '.weight'])
        # self.init_data = torch.stack(self.init_data, dim=0)
        # self.weight.data = self.init_data
        # if self.use_bias:
        #     self.init_data_bias.append(state_dicts[-1][self.name + '.bias'])
        #     self.init_data_bias = torch.stack(self.init_data_bias, dim=0)
        #     self.bias.data = self.init_data_bias
        self.disable_scaling = disable_scaling
        self.num_experts = len(state_dicts)
        self.average_route = average_route
        init_data_a = []
        init_data_b = []
        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1,1))
        if average_route:
            self._routing_fn_a = lambda in_feature: torch.empty((in_feature.size(0), self.num_experts),
                                                                device=in_feature.device).fill_(1 / self.num_experts)
            self._routing_fn_b = lambda in_feature: torch.empty((in_feature.size(0), self.num_experts),
                                                                device=in_feature.device).fill_(1 / self.num_experts)
        else:
            self._routing_fn_a = _routing(self.conv.in_channels, self.num_experts, self.router_drop_out, act=route_act)
            self._routing_fn_b = _routing(self.conv.in_channels, self.num_experts, self.router_drop_out, act=route_act)


        self.lora_A_weight = nn.Parameter(self.conv.weight.new_zeros((self.num_experts,
                                                                      int(self.r * self.conv.out_channels) * self.conv.kernel_size[0], self.conv.in_channels * self.conv.kernel_size[0])),
                                          requires_grad=not frozen_experts)
        self.lora_B_weight = nn.Parameter(self.conv.weight.new_zeros((self.num_experts, self.conv.out_channels // self.conv.groups * self.conv.kernel_size[0], int(self.r * self.conv.out_channels) * self.conv.kernel_size[1])),
                                          requires_grad=not frozen_experts)
        for i, state_dict in enumerate(state_dicts):
            #
            init_data_a.append(state_dict[self.name + '.lora_A'])
            init_data_b.append(state_dict[self.name + '.lora_B'])

        init_data_a = torch.stack(init_data_a, dim=0)
        init_data_b = torch.stack(init_data_b, dim=0)

        self.lora_A_weight.data.copy_(init_data_a)
        self.lora_B_weight.data.copy_(init_data_b)

        assert not disable_scaling

    def forward(self, x, mode='all'):

        if mode == 'all':
            if self.enable_lora:

                ndim = x.ndim
                if ndim == 4:
                    bsz, _, _, _ = x.shape
                    assert bsz == 1
                    # 标准2D
                    pooled_inputs = self._avg_pooling(x).flatten(start_dim=1)  # assert
                    if self.num_experts == 1:
                        routing_weights_a = self._routing_fn_a(pooled_inputs).squeeze(0) # [num_exp]
                        # print("routing_weights_a", routing_weights_a.shape)
                        routing_weights_b = self._routing_fn_b(pooled_inputs).squeeze(0)
                    else:
                        routing_weights_a = self._routing_fn_a(pooled_inputs).squeeze()  # [num_exp]
                        routing_weights_b = self._routing_fn_b(pooled_inputs).squeeze()

                    lora_A = torch.sum(routing_weights_a[:, None, None] * self.lora_A_weight, 0)
                    lora_B = torch.sum(routing_weights_b[:,None, None] * self.lora_B_weight, 0)
                else:
                    raise NotImplementedError
                return self.conv._conv_forward(
                    x,
                    self.conv.weight + (lora_B @ lora_A).view(self.conv.weight.shape) * self.scaling,
                    self.conv.bias
                )
            else:
                return self.conv._conv_forward(
                    x,
                    self.conv.weight,
                    self.conv.bias
                )
        elif mode == 'sigma':
            raise NotImplementedError
            return self.conv(x)
        elif mode == 'tau':
            raise NotImplementedError
            return self.conv._conv_forward(
                x,
                (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        else:
            raise NotImplementedError

    def extra_repr(self) -> str:
        return (super().extra_repr()) + "," + f"rank={self.r},alpha={self.lora_alpha},num_experts={self.num_experts},disable_scaling={self.disable_scaling},enable_lora={self.enable_lora}"

class MoEConv2DLoRA(MoEConvNDLoRA):
    def __init__(self, *args, **kwargs):
        super(MoEConv2DLoRA, self).__init__(nn.Conv2d, *args, **kwargs)

if __name__ == '__main__':
    s = torch.tensor([[0.56],[0.1]])
    j = F.softmax(s, dim=1)
    f =4