import copy
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


class clientOur(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        gen_shapes = []
        gen_param_names = []
        for module in self.model.modules():
            if isinstance(module, FunctionalLoRALinear):
                linear_layer: nn.Linear = module.original_layer
                gen_shapes.append((linear_layer.in_features, module.rank))
                gen_shapes.append((module.rank, linear_layer.out_features))
                gen_param_names.append(f"{module.name}.lora_A")
                gen_param_names.append(f"{module.name}.lora_B")
        feature_dim = 0
        if self.model_str.startswith('ab_mil'):
            feature_dim = 512
        self.hypernet = Prototype2LoRA_Hypernet(feature_dim, gen_shapes)
        self.gen_param_names = gen_param_names

        if not self.save_gpu:
            self.hypernet = self.hypernet.to(self.device)


    def reset_optimizer(self, rnd, is_latest_finetune=False):
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

        self.optimizer = create_optimizer(self.model.parameters())
        self.hyper_optimizer = create_optimizer(self.hypernet.parameters(), lr_multi=10.0)


    def train(self, rnd, is_latest_finetune=False, **kwargs):
        self.reset_optimizer(rnd, is_latest_finetune=is_latest_finetune)
        train_dataloader = self.load_train_data()
        if self.save_gpu:
            self.model.to(self.device)
            self.hypernet.to(self.device)

        self.model.train()
        self.hypernet.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):

            loss_task = 0.0
            loss_lora = 0.0
            for i, batch_dict in enumerate(train_dataloader):
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
                # 但是不需要输入lora作为参数
                feature_dict = self.forward_base(batch_dict)
                output_dict = self.forward_head(batch_dict, feature_dict)
                loss = output_dict['loss']
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                self.optimizer.step()
                loss_task += loss.item()


            for i, batch_dict in enumerate(train_dataloader):
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
                # 但是不需要输入lora作为参数
                feature_dict = self.forward_base(batch_dict)
                # 输入
                prototype_gen = self.hypernet(feature_dict["M"].squeeze(0).detach())
                gen_param_state_dict = {k: v for k, v in zip(self.gen_param_names, prototype_gen)}
                feature_dict = self.forward_base(batch_dict, additional_param_dict=gen_param_state_dict)
                output_dict = self.forward_head(batch_dict, feature_dict)
                loss = output_dict['loss']
                self.hyper_optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                self.hyper_optimizer.step()
                loss_lora += loss.item()
            loss_task = loss_task / len(train_dataloader)
            loss_lora = loss_lora / len(train_dataloader)




        if self.save_gpu:
            self.model.cpu()
            self.hypernet.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        return loss_task, loss_lora

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
        self.hypernet.eval()
        if self.save_gpu:
            self.model.to(self.device)
            self.hypernet.to(self.device)

        test_num = 0

        with torch.no_grad():
            for batch_dict in testloaderfull:
                batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items() }
                # output_dict = self.forward_model(batch_dict)
                # loss = output_dict['loss']

                feature_dict = self.forward_base(batch_dict)
                # output_dict = self.forward_head(batch_dict, feature_dict)
                prototype_gen = self.hypernet(feature_dict["M"].squeeze(0).detach())
                gen_param_state_dict = {k: v for k, v in zip(self.gen_param_names, prototype_gen)}
                feature_dict = self.forward_base(batch_dict, additional_param_dict=gen_param_state_dict)
                output_dict = self.forward_head(batch_dict, feature_dict)
                loss = output_dict['loss']

                self.update_metrics(output_dict['logits'], batch_dict, data_split=data_split, is_train=False, loss=loss)

                test_num += batch_dict['feature'].shape[0]  # dim0是具体的sample数量. dim1是patch级别的数量

        if self.save_gpu:
            self.model.cpu()
            self.hypernet.cpu()
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

