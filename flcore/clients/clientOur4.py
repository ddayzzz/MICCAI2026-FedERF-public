import copy
import os.path
from collections import OrderedDict
from typing import Literal

import torch
import numpy as np
import time
import faiss
from torch import nn as nn, optim

from flcore.clients.clientbase import Client, get_fl_lr
from flcore.trainmodel.lora_modules.lora_utils import mark_only_lora_as_trainable, mark_only_weight_as_trainable

class clientOur4(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.lora_lr = 1e-3
        self.lora_wd = 1e-2

    def set_parameters(self, model: nn.Module, state_dict=False):
        for (new_name, new_param), (old_name, old_param) in zip(model.state_dict().items(), self.model.state_dict().items()):
            if "lora_" in new_name:
                continue
            old_param.data.copy_(new_param.data)
    #
    # def reset_optimizer(self, rnd,):
    #     def create_optimizer(params, lr_multi=1.0):
    #         if self.lr_decay is not None:
    #             lr_init = get_fl_lr(rnd, total_rounds=self.global_rounds,
    #                                 base_lr=self.learning_rate, min_lr=self.local_learning_rate_end,
    #                                 mode=self.lr_decay, warmup_rounds=self.lr_warmup_round)
    #         else:
    #             lr_init = self.learning_rate
    #         lr_init *= lr_multi
    #         # print("lr_init: {}".format(lr_init))
    #         if self.optimizer_name == "adam":
    #             optimizer = torch.optim.Adam(params,
    #                                               lr=lr_init, weight_decay=self.weight_decay,
    #                                               betas=(0.9, 0.999), )
    #         elif self.optimizer_name == "adamw":
    #             optimizer = torch.optim.AdamW(params,
    #                                                lr=lr_init, weight_decay=self.weight_decay,
    #                                                betas=(0.9, 0.999), )
    #         elif self.optimizer_name == "sgd":
    #             optimizer = torch.optim.SGD(params,
    #                                              lr=lr_init, weight_decay=self.weight_decay, momentum=0.9)
    #         else:
    #             raise NotImplementedError("Unsupported optimizer name: %s" % self.optimizer_name)
    #         return optimizer
    #
    #     weights = dict(self.model.named_parameters())
    #     lora_weights, non_lora_weights = [], []
    #     for k in weights.keys():
    #         if 'lora_A' in k or 'lora_B' in k:
    #             lora_weights.append(weights[k])
    #         else:
    #             non_lora_weights.append(weights[k])
    #     self.optimizer = create_optimizer(non_lora_weights, lr_multi=1.0)
    #     self.lora_optimizer = create_optimizer(lora_weights, lr_multi=1.0)

    # def train(self, rnd, is_latest_finetune=False, gen_lora_params: OrderedDict = None, **kwargs):
    #     self.reset_optimizer(rnd, is_latest_finetune=is_latest_finetune)
    #     train_dataloader = self.load_train_data()
    #     if self.save_gpu:
    #         self.model.to(self.device)
    #
    #     self.model.train()
    #
    #     start_time = time.time()
    #
    #     max_local_epochs = self.local_epochs
    #     if self.train_slow:
    #         max_local_epochs = np.random.randint(1, max_local_epochs // 2)
    #     lora_local_epochs = 1
    #
    #     # if gen_lora_params is not None:
    #     #     with torch.no_grad():
    #     #         self.model.load_state_dict(gen_lora_params, strict=False)
    #
    #     # 本地计算
    #     self.model.model.set_lora_enabled(False)  #关闭 LoRA的额外 分支
    #     self.model.train()
    #     mark_only_weight_as_trainable(self.model)
    #     for epoch in range(max_local_epochs - lora_local_epochs):
    #
    #         loss_task = 0.0
    #         for i, batch_dict in enumerate(train_dataloader):
    #             if self.train_slow:
    #                 time.sleep(0.1 * np.abs(np.random.rand()))
    #             batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
    #
    #             output_dict = self.forward_model(batch_dict)
    #             loss = output_dict['loss']
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             if self.grad_clip_max_norm is not None:
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
    #             self.optimizer.step()
    #             loss_task += loss.item()
    #
    #         loss_task = loss_task / len(train_dataloader)
    #     # lora
    #     mark_only_lora_as_trainable(self.model)
    #     self.model.model.set_lora_train_while_rest_is_mode(False)
    #     self.model.model.set_lora_enabled(True)  # 开启 LoRA的额外 分支
    #     for epoch in range(lora_local_epochs):
    #
    #         loss_lora = 0.0
    #         for i, batch_dict in enumerate(train_dataloader):
    #             if self.train_slow:
    #                 time.sleep(0.1 * np.abs(np.random.rand()))
    #             batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
    #
    #             output_dict = self.forward_model(batch_dict)
    #             loss = output_dict['loss']
    #             self.lora_optimizer.zero_grad()
    #             loss.backward()
    #             if self.grad_clip_max_norm is not None:
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
    #             self.lora_optimizer.step()
    #             loss_lora += loss.item()
    #
    #         loss_lora = loss_lora / len(train_dataloader)
    #
    #     if self.save_gpu:
    #         self.model.cpu()
    #
    #     self.train_time_cost['num_rounds'] += 1
    #     self.train_time_cost['total_cost'] += time.time() - start_time
    #
    #     return loss_task, loss_lora
    # 26-02-06 15:37 改为lora最后几个epoch进行finetune!
    def train(self, rnd, ):
        self.reset_optimizer(rnd)
        train_dataloader = self.load_train_data()
        if self.save_gpu:
            self.model.to(self.device)

        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)


        # if gen_lora_params is not None:
        #     with torch.no_grad():
        #         self.model.load_state_dict(gen_lora_params, strict=False)

        # 本地计算
        if self.model_str.startswith("mico"):
            self.model.set_lora_enabled(False)  #关闭 LoRA的额外 分支
        else:
            self.model.model.set_lora_enabled(False)  # 关闭 LoRA的额外 分支
        mark_only_weight_as_trainable(self.model)
        for epoch in range(max_local_epochs):

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

            loss_task = loss_task / len(train_dataloader)

        if self.save_gpu:
            self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        return loss_task


    def finetune_lora(self, lora_local_epochs=None):
        if lora_local_epochs is None:
            lora_local_epochs = self.local_epochs * self.global_rounds
        train_dataloader = self.load_train_data()
        # lora
        mark_only_lora_as_trainable(self.model)
        if self.model_str.startswith("mico"):
            self.model.set_lora_train_while_rest_is_mode(False)
            self.model.set_lora_enabled(True)  # 开启 LoRA的额外 分支
        else:
            self.model.model.set_lora_train_while_rest_is_mode(False)
            self.model.model.set_lora_enabled(True)  # 开启 LoRA的额外 分支
        lora_names = []
        lora_weights = []
        # last epoch

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                lora_names.append(n)
                lora_weights.append(p)
        print(
            f"Last round, finetune {lora_local_epochs} epochs, lr_init={self.lora_lr}, wd={self.lora_wd}, opt=adamw, scheduler=cosine, para names: {lora_names}")
        lora_optimizer = optim.AdamW(lora_weights, lr=self.lora_lr, weight_decay=self.lora_wd, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lora_optimizer, T_max=lora_local_epochs, eta_min=1e-6)
        for epoch in range(lora_local_epochs):

            loss_lora = 0.0
            for i, batch_dict in enumerate(train_dataloader):
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in
                              batch_dict.items()}

                output_dict = self.forward_model(batch_dict)
                loss = output_dict['loss']
                lora_optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)

                lora_optimizer.step()
                loss_lora += loss.item()

            loss_lora = loss_lora / len(train_dataloader)
            scheduler.step()
        print(f"Site[{self.train_client_names[self.id]}], lora_loss: {loss_lora}")
        return loss_lora