import copy
from collections import OrderedDict
from typing import Literal

import torch
import numpy as np
import time

from flcore.trainmodel.lora_modules.lora_layers import Linear as LoRALinearLayer, FunctionalLinear
from torch import nn

from flcore.clients.clientbase import Client, get_fl_lr
import os
import torch.nn.functional as F

from flcore.trainmodel.hypernet2 import ViTHyper
from flcore.trainmodel.route_modules import ProtoGuidedLoRA
from flcore.trainmodel.lora_modules.lora_utils import mark_only_lora_as_trainable, mark_only_weight_as_trainable

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(torch.numel(param))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


class clientOur3(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # self.cached_lora_parameters = {k: v for k, v in self.model.named_parameters()}
        self.hyper_lr = 0.001
        self.hyper_opt_name = "adamw"
        self.hyper_wd = 0.00001
        self.Budget = []
        lora_names = []
        lora_shapes = []
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinearLayer):
                lora_names.append(name)
                lora_shapes.append((tuple(module.lora_A.size()), tuple(module.lora_B.size())))
            elif isinstance(module, FunctionalLinear):
                lora_names.append(name)
                lora_shapes.append(((module.r, module.in_features), (module.out_features,module.r)))

        hyper = ViTHyper(embedding_dim=384, embedding_hidden_dim=256, hidden_dim=384, layer_shapes=lora_shapes,
                         layer_lora_names_prefixs=lora_names, att_hidden_dim=256)
        self.hyper = hyper.to(self.device)




    def reset_optimizer(self, rnd, is_latest_finetune=False, ):
        def create_optimizer(params, lr_multi=1.0):
            if is_latest_finetune:
                lr_init = self.learning_rate
            else:
                if self.lr_decay is not None:
                    lr_init = get_fl_lr(rnd, total_rounds=self.global_rounds,
                                        base_lr=self.learning_rate, min_lr=self.local_learning_rate_end,
                                        mode=self.lr_decay, warmup_rounds=self.lr_warmup_round)

                else:
                    lr_init = self.learning_rate
            lr_init *= lr_multi
            # print("lr_init: {}".format(lr_init))
            if self.optimizer_name == "adam":
                optimizer = torch.optim.Adam(params,
                                                  lr=lr_init, weight_decay=self.weight_decay,
                                                  betas=(0.9, 0.999), )
            elif self.optimizer_name == "adamw":
                optimizer = torch.optim.AdamW(params,
                                                   lr=lr_init, weight_decay=self.weight_decay,
                                                   betas=(0.9, 0.999), )
            elif self.optimizer_name == "sgd":
                optimizer = torch.optim.SGD(params,
                                                 lr=lr_init, weight_decay=self.weight_decay, momentum=0.9)
            else:
                raise NotImplementedError("Unsupported optimizer name: %s" % self.optimizer_name)
            return optimizer

        # weights = dict(self.model.named_parameters())
        # lora_weights, non_lora_weights = [], []
        # for k in weights.keys():
        #     if 'lora_A' in k or 'lora_B' in k:
        #         lora_weights.append(weights[k])
        #     else:
        #         non_lora_weights.append(weights[k])
        self.optimizer = create_optimizer(self.model.parameters(), lr_multi=1.0)
        # self.lora_optimizer = create_optimizer(lora_weights, lr_multi=1.0)
        if self.hyper_opt_name == "adamw":
            self.hyper_optimizer = torch.optim.AdamW(self.hyper.parameters(), lr=self.hyper_lr, weight_decay=self.hyper_wd)


    # def set_parameters(self, model: nn.Module, sync_lora, state_dict=False):
    #     for new_param, (pname, old_param) in zip(self.get_trainable_parameters(model, state_dict=state_dict),
    #                                     self.get_trainable_parameters(self.model, state_dict=state_dict, return_names=True).items()):
    #         # old_param.data = new_param.data.clone()
    #         # print("src:",ks[i], new_param.shape, ", desc: ", ss[i],old_param.shape)
    #         if not sync_lora:
    #             if "lora_A" in pname or "lora_B" in pname:
    #                 continue  # 个性化的部分
    #
    #         old_param.data.copy_(new_param.data)
    #         # i += 1

    def train(self, rnd, is_latest_finetune=False, **kwargs):
        self.reset_optimizer(rnd, is_latest_finetune=is_latest_finetune)
        train_dataloader = self.load_train_data()
        if self.save_gpu:
            self.model.to(self.device)
            self.hyper.to(self.device)

        self.model.train()
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        lora_local_epochs = 1

        # backup_global_hyper_vec = copy.deepcopy(get_flat_params_from(self.hyper))

        # 本地计算
        # mark_only_weight_as_trainable(self.model)
        # for epoch in range(max_local_epochs - lora_local_epochs):
        for epoch in range(max_local_epochs):

            loss_task = 0.0
            for i, batch_dict in enumerate(train_dataloader):
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}


                patched_emb = self.model.model.forward_encode_features(batch_dict['feature'])
                #
                lora_weights = self.hyper(patched_emb)
                output_dict = self.forward_model(batch_dict, state_dict=lora_weights, patched_emb=patched_emb)
                loss = output_dict['loss']
                self.optimizer.zero_grad()
                self.hyper_optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                self.optimizer.step()
                self.hyper_optimizer.step()
                loss_task += loss.item()
                # if i >= 1:
                #     break

            loss_task = loss_task / len(train_dataloader)






        if self.save_gpu:
            self.model.cpu()
            self.hyper.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # self.slide_features = (slide_features / n_cnt).squeeze()
        updated_hyper_vec = copy.deepcopy(get_flat_params_from(self.hyper))
        # self.pseudo_hyper_gradient = backup_global_hyper_vec - updated_hyper_vec
        self.pseudo_hyper_gradient = updated_hyper_vec
        return loss_task, 0.0

    def test_metrics(self, data_split: Literal['train', 'val', 'test'], **load_data_kwargs):
        if data_split == "train":
            testloaderfull = self.load_train_data(**load_data_kwargs)
        elif data_split == "val":
            testloaderfull = self.load_val_data(**load_data_kwargs)
        elif data_split == "test":
            testloaderfull = self.load_test_data(**load_data_kwargs)
        else:
            raise NotImplementedError
        self.model.eval()
        if self.save_gpu:
            self.model.to(self.device)

        test_num = 0

        with torch.no_grad():
            for batch_dict in testloaderfull:
                batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items() }
                # output_dict = self.forward_model(batch_dict)
                patched_emb = self.model.model.forward_encode_features(batch_dict['feature'])
                #
                lora_weights = self.hyper(patched_emb)
                output_dict = self.forward_model(batch_dict, state_dict=lora_weights, patched_emb=patched_emb)

                loss = output_dict['loss']
                self.update_metrics(output_dict['logits'], batch_dict, data_split=data_split, is_train=False, loss=loss)

                test_num += batch_dict['feature'].shape[0]  # dim0是具体的sample数量. dim1是patch级别的数量

        if self.save_gpu:
            self.model.cpu()
            # self.generator.cpu()
        return test_num, self.describe_metrics(data_split=data_split)

    @torch.no_grad()
    def collect_prototype(self):
        train_dataloader = self.load_train_data()
        if self.save_gpu:
            self.model.to(self.device)

        self.model.eval()
        protos = None
        count = 0
        for i, batch_dict in enumerate(train_dataloader):
            batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
            # 但是不需要输入lora作为参数
            feature_dict = self.forward_base(batch_dict)
            f = feature_dict["M"].squeeze(0)  # [n_dim]
            count += 1
            if protos is None:
                protos = f
            else:
                protos += f
        return protos.cpu() / count


def compute_R(A_loc_list, A_glb_list, B_loc_list, B_glb_list):
    # mslora的定义是反着的,a: r x dim, b: dim x r
    R = 0.0

    for a_loc, a_glb, b_loc, b_glb in zip(A_loc_list, A_glb_list, B_loc_list, B_glb_list):
        Ia = torch.eye(a_loc.size(1), device=a_loc.device, dtype=a_loc.dtype)
        Ib = torch.eye(b_glb.size(0), device=a_glb.device, dtype=a_glb.dtype)
        PP_T = torch.matmul(a_loc.T, a_glb)
        QQ_T = torch.matmul(b_loc, b_glb.T)
        R += torch.norm(PP_T - Ia, p='fro') ** 2 + torch.norm(QQ_T - Ib, p='fro') ** 2
    R /= len(A_loc_list)
    return R

def compute_orth_loss(lora_model, global_lora_params: OrderedDict):
    a_loc_params = []
    a_glb_params = []
    b_loc_params = []
    b_glb_params = []
    for k, v in lora_model.named_parameters():
        if "lora_A" in k:
            a_loc_params.append(v)
            a_glb_params.append(global_lora_params[k])
        if "lora_B" in k:
            b_loc_params.append(v)
            b_glb_params.append(global_lora_params[k])
    loss = compute_R(a_loc_params, a_glb_params, b_loc_params, b_glb_params)  #
    return loss


