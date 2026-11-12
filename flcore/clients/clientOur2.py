import copy
from collections import OrderedDict
from typing import Literal

import torch
import numpy as np
import time

from torch import nn

from flcore.clients.clientbase import Client, get_fl_lr
import os
import torch.nn.functional as F

from flcore.trainmodel.hypernet import FunctionalLoRALinear, Prototype2LoRA_Hypernet
from flcore.trainmodel.route_modules import ProtoGuidedLoRA
from flcore.trainmodel.lora_modules.lora_utils import mark_only_lora_as_trainable, mark_only_weight_as_trainable


class clientOur2(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # self.cached_lora_parameters = {k: v for k, v in self.model.named_parameters()}


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

        weights = dict(self.model.named_parameters())
        lora_weights, non_lora_weights = [], []
        for k in weights.keys():
            if 'lora_A' in k or 'lora_B' in k:
                lora_weights.append(weights[k])
            else:
                non_lora_weights.append(weights[k])
        self.optimizer = create_optimizer(non_lora_weights, lr_multi=1.0)
        self.lora_optimizer = create_optimizer(lora_weights, lr_multi=1.0)


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

    def train(self, rnd, is_latest_finetune=False, gen_lora_params: OrderedDict = None, **kwargs):
        self.reset_optimizer(rnd, is_latest_finetune=is_latest_finetune)
        train_dataloader = self.load_train_data()
        if self.save_gpu:
            self.model.to(self.device)

        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        lora_local_epochs = 1

        if gen_lora_params is not None:
            with torch.no_grad():
                self.model.load_state_dict(gen_lora_params, strict=False)
        # 本地计算
        mark_only_weight_as_trainable(self.model)
        for epoch in range(max_local_epochs - lora_local_epochs):

            loss_task = 0.0
            for i, batch_dict in enumerate(train_dataloader):
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}

                output_dict = self.forward_model(batch_dict)
                loss = output_dict['loss']
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                self.optimizer.step()
                loss_task += loss.item()
                # if i >= 1:
                #     break

            loss_task = loss_task / len(train_dataloader)

        # lora
        slide_features = None
        n_cnt = 0
        mark_only_lora_as_trainable(self.model)
        is_first = True
        for epoch in range(lora_local_epochs):

            loss_lora = 0.0
            for i, batch_dict in enumerate(train_dataloader):
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}

                output_dict = self.forward_model(batch_dict)
                loss = output_dict['loss']
                if not is_first:
                    # 计算
                    if gen_lora_params is not None:
                        # 计算和全局lora的差值
                        orth_loss = compute_orth_loss(self.model, global_lora_params=gen_lora_params)
                        loss += (0.01 * orth_loss)
                self.lora_optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                self.lora_optimizer.step()
                loss_lora += loss.item()

                if slide_features is None:
                    slide_features = output_dict['slide_feats'].detach().clone()
                else:
                    slide_features += (output_dict['slide_feats'].detach().clone())
                n_cnt += 1

                is_first = False

                # if i >= 1:
                #     break

            loss_lora = loss_lora / len(train_dataloader)




        if self.save_gpu:
            self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        self.slide_features = (slide_features / n_cnt).squeeze()

        return loss_task, loss_lora


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


