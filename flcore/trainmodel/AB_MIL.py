import functools

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from flcore.trainmodel.hypernet import FunctionalLoRALinear, MoEFunctionalLoRA
from flcore.trainmodel.route_modules import RouteLinear, ProtoGuidedLoRA
from flcore.trainmodel.lora_modules.lora_layers import LoRALayer, Linear


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class AB_MIL(nn.Module):
    def __init__(self, L=512, D=128, num_classes=2, dropout=0, act=nn.ReLU(), in_dim=512, rrt=None):
        super(AB_MIL, self).__init__()
        self.rrt = rrt
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.L = L
        self.D = D
        self.K = 1
        self.feature = [nn.Linear(in_dim, self.L)]

        self.feature += [act]

        if dropout:
            self.feature += [nn.Dropout(dropout)]

        if self.rrt != None:
            self.feature += [self.rrt]

        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.num_classes),
        )

        self.apply(initialize_weights)

    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        feature = self.feature(x)
        # feature = group_shuffle(feature)
        feature = feature.squeeze(0)
        A = self.attention(feature)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # 1,KxL
        logits = self.classifier(M)
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = M
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_ori
        return forward_return

class AB_MIL_IOP_FLRoute(nn.Module):

    def __init__(self, L=512, D=128, num_classes=2, dropout=0, act=nn.ReLU(), in_dim=512, rrt=None):
        super(AB_MIL_IOP_FLRoute, self).__init__()
        self.rrt = rrt
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.L = L
        self.D = D
        self.K = 1
        # self.feature = [nn.Linear(in_dim, self.L)]
        self.feature = [RouteLinear(in_dim, self.L, name="feature")]

        self.feature += [act]

        if dropout:
            self.feature += [nn.Dropout(dropout)]

        if self.rrt != None:
            self.feature += [self.rrt]

        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            # nn.Linear(self.L, self.D),
            RouteLinear(self.L, self.D, name="attention.0"),
            nn.Tanh(),
            # nn.Linear(self.D, self.K)
            RouteLinear(self.D, self.K, name="attention.2")
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.num_classes),
        )

        self.apply(initialize_weights)

    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        feature = self.feature(x)
        # feature = group_shuffle(feature)
        feature = feature.squeeze(0)
        A = self.attention(feature)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # 1,KxL
        logits = self.classifier(M)
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = M
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_ori
        return forward_return

    def reinit_trajectories(self,state_dicts):
        client_names = [k for k in state_dicts.keys() if k != 'global']
        for m in self.modules():
            if isinstance(m, RouteLinear):
                m._mix_trajectories(state_dicts, client_names=client_names)


class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(in_channels, 2 * in_channels)
        self.fc2 = nn.Linear(2 * in_channels, num_experts)

    def forward(self, x):
        x = torch.flatten(x)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.sigmoid(x)

class AB_MIL_with_LoRA(nn.Module):
    def __init__(self, L=512, D=128, num_classes=2, dropout=0, act=nn.ReLU(), in_dim=512, rrt=None,
                 linear_r=8,
                 add_router=False):
        super(AB_MIL_with_LoRA, self).__init__()
        self.rrt = rrt
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.L = L
        self.D = D
        self.K = 1
        self.add_router = add_router
        # self.feature = [ProtoGuidedLoRA(in_dim, self.L, r=linear_r, num_prototypes=4),]  # TODO 这里写死了, 是4个客户端
        lora_fn = lambda i_dim,out_dim,name: Linear(i_dim, out_dim, r=linear_r)
        norm_linear_fn = lambda i_dim,out_dim,name: nn.Linear(i_dim, out_dim)
        if add_router:
            lora_fn = lambda i_dim,out_dim, name: ProtoGuidedLoRA(i_dim, out_dim, r=linear_r, name=name)
        self.feature = [norm_linear_fn(in_dim, self.L, name="feature.0"), ]

        self.feature += [act]

        if dropout:
            self.feature += [nn.Dropout(dropout)]

        if self.rrt != None:
            self.feature += [self.rrt]

        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            # ProtoGuidedLoRA(self.L, self.D, r=linear_r, num_prototypes=4),
            norm_linear_fn(self.L, self.D, name="attention.0"),
            # lora_fn_debug(self.L, self.D, name="attention.0"),
            nn.Tanh(),
            # ProtoGuidedLoRA(self.D, self.K, r=linear_r, num_prototypes=4)
            norm_linear_fn(self.D, self.K, name="attention.2"),
            # lora_fn_debug(self.D, self.K, name="attention.2"),
        )
        self.classifier = nn.Sequential(
            # nn.Linear(self.L * self.K, self.num_classes,),
            lora_fn(self.L * self.K, self.num_classes, name="classifier.0"),
            # lora_fn_debug(self.L * self.K, self.num_classes, name="classifier.0"),
        )

        self.apply(initialize_weights)

    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        feature = self.feature(x)
        # feature = group_shuffle(feature)
        feature = feature.squeeze(0)
        A = self.attention(feature)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # 1,KxL
        logits = self.classifier(M)
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = M
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_ori
        return forward_return

    def forward_feature(self, x):
        forward_return = {}
        feature = self.feature(x)
        # feature = group_shuffle(feature)
        feature = feature.squeeze(0)
        A = self.attention(feature)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # 1,KxL
        return {'M': M}

    def forward_head(self, feature_dict, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = dict()
        M = feature_dict['M']
        logits = self.classifier(M)
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = M
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_ori
        return forward_return

class AB_MIL_with_LoRA2(nn.Module):
    def __init__(self, L=512, D=128, num_classes=2, dropout=0, act=nn.ReLU(), in_dim=512, rrt=None,
                 linear_r=8,
                 alpha=16,
                 add_router=False):
        super(AB_MIL_with_LoRA2, self).__init__()
        self.rrt = rrt
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.L = L
        self.D = D
        self.K = 1
        self.add_router = add_router
        lora_fn = lambda i_dim,out_dim,name: FunctionalLoRALinear(original_layer=nn.Linear(i_dim, out_dim), alpha=alpha, rank=linear_r, name=name)
        norm_linear_fn = lambda i_dim,out_dim,name: nn.Linear(i_dim, out_dim)
        if add_router:
            lora_fn = lambda i_dim,out_dim, name: MoEFunctionalLoRA(original_layer=nn.Linear(i_dim, out_dim), rank=linear_r, alpha=alpha, name=name)
        self.feature = [lora_fn(in_dim, self.L, name="feature.0"), ]

        self.feature += [act]

        if dropout:
            self.feature += [nn.Dropout(dropout)]

        if self.rrt != None:
            self.feature += [self.rrt]

        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            # ProtoGuidedLoRA(self.L, self.D, r=linear_r, num_prototypes=4),
            # norm_linear_fn(self.L, self.D, name="attention.0"),
            lora_fn(self.L, self.D, name="attention.0"),
            nn.Tanh(),
            # ProtoGuidedLoRA(self.D, self.K, r=linear_r, num_prototypes=4)
            # norm_linear_fn(self.D, self.K, name="attention.2"),
            lora_fn(self.D, self.K, name="attention.2"),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.num_classes,),
            # lora_fn(self.L * self.K, self.num_classes, name="classifier.0"),
            # lora_fn(self.L * self.K, self.num_classes, name="classifier.0"),
        )

        self.apply(initialize_weights)


    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False, additional_param_dict=None):
        forward_return = {}
        # feature = self.feature(x)
        # # feature = group_shuffle(feature)
        # feature = feature.squeeze(0)
        # A = self.attention(feature)
        # A_ori = A.clone()
        # A = torch.transpose(A, -1, -2)  # KxN
        # A = F.softmax(A, dim=-1)  # softmax over N
        # M = torch.mm(A, feature)  # 1,KxL
        M = self.forward_feature(x, additional_param_dict)['M']
        logits = self.classifier(M)
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = M
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_ori
        return forward_return

    def forward_feature(self, x, additional_param_dict=None):
        forward_return = {}
        # feature = self.feature(x)
        feature = x
        for feature_layer in self.feature:
            if isinstance(feature_layer, FunctionalLoRALinear):
                feature = feature_layer(feature, additional_param_dict)
            else:
                feature = feature_layer(feature)

        # feature = group_shuffle(feature)
        feature = feature.squeeze(0)
        # A = self.attention(feature)
        A = feature
        for att_layer in self.attention:
            if isinstance(att_layer, FunctionalLoRALinear):
                A = att_layer(A, additional_param_dict)
            else:
                A = att_layer(A)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # 1,KxL
        return {'M': M}

    def forward_head(self, feature_dict, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = dict()
        M = feature_dict['M']
        logits = self.classifier(M)
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = M
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_ori
        return forward_return


if __name__ == '__main__':
    # st = "/xxx/projects2/fedmed2026/models/data_tcga_model_ab_mil_lora/eval_site_LIHC/Our_glb20_loc1_lr0.0002_wd0.00001_optadam_nc4jr1.0_bs1_server_fold0.pt"
    # ck = torch.load(st, map_location="cpu")
    # model = AB_MIL()
    #== pretrain
    # model = AB_MIL_with_LoRA(add_router=False, num_classes=4)
    #===
    #=== finetune
    model = AB_MIL_with_LoRA2(add_router=False, num_classes=4)

    # lora_state_dicts = []
    #
    # for client_name, sub_state_dict in ck['backup_recent_lora_params'].items():
    #     lora_state_dicts.append(sub_state_dict)
    # lora_state_dicts.append(ck['global_model'])
    # for module in model.modules():
    #     if isinstance(module, ProtoGuidedLoRA):
    #         module._mix_trajectories(lora_state_dicts)
    #===
    x = torch.randn(1, 40, 512)
    y = model(x)
    f = 4