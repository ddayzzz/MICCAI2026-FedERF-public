import copy
from collections import OrderedDict
from typing import Optional, Union, Iterator, Dict, Tuple, Literal

import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
import torchmetrics
# from adeval import EvalAccumulator
# from scipy.ndimage import gaussian_filter
# from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchmetrics import Accuracy

from flcore.trainmodel.AB_MIL import AB_MIL, AB_MIL_with_LoRA, AB_MIL_with_LoRA2
from flcore.trainmodel.WIKG_MIL import WIKG_MIL
from flcore.trainmodel.generator.autoencoder import MultimodalAutoencoder, get_autoencoder
from flcore.trainmodel.mico import MiCo
from flcore.trainmodel.mil_models.abmil import ABMILModel
from flcore.trainmodel.mil_models.abmil2 import ABMILModel2
from flcore.trainmodel.mil_models.mil_template import MIL
from flcore.trainmodel.mil_models.transmil import TransMIL, TransMILModel
from flcore.trainmodel.mil_models.wikg_mil import WIKGMILModel
from flcore.trainmodel.mm_models import ImageTextClassifier
from flcore.trainmodel.my_metrics import Scalar, AUROC, C_Index
from flcore.trainmodel.splitmodalnet import SplitModalNet
from flcore.trainmodel.trans_mil.trans_mil import TRANS_MIL
from flcore.trainmodel.trans_mil.trans_mil2 import TransMIL2
from flcore.utils.evaluator import EvalMetric
from flcore.utils.read_client_data import read_client_data
import math

from flcore.utils.survial_loss import NLLSurvLoss


def get_fl_lr(current_round, total_rounds, base_lr=2e-4, min_lr=1e-6, warmup_rounds=0, mode='cosine'):
    """
    无状态学习率计算函数，专为联邦学习设计。

    参数:
        current_round (int): 当前全局通信轮次 (0 到 total_rounds-1)
        total_rounds (int): 总的通信轮次
        base_lr (float): 初始学习率 (最大值)
        min_lr (float): 训练结束时的最小学习率
        warmup_rounds (int): 全局预热轮数
        mode (str): 'cosine' 或 'linear'

    返回:
        lr (float): 当前轮次应使用的学习率
    """

    # 1. Global Warmup 阶段 (可选)
    # 在联邦学习初期，为了防止模型权重剧烈波动导致聚合失效，
    # 有时需要前几轮由 0 慢慢增加到 base_lr
    if current_round < warmup_rounds:
        return base_lr * (current_round + 1) / max(1, warmup_rounds)

    # 2. Decay 阶段
    # 计算从 warmup 结束到训练结束的进度 (0.0 -> 1.0)
    progress = (current_round - warmup_rounds) / max(1, total_rounds - warmup_rounds)

    # 确保进度不超过 1.0
    progress = min(1.0, max(0.0, progress))

    if mode == 'cosine':
        # 余弦衰减: 0.5 * (1 + cos(pi * progress)) 从 1 降到 0
        decay_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif mode == 'linear':
        decay_factor = 1.0 - progress
    else:
        decay_factor = 1.0

    # 3. 映射到 [min_lr, base_lr] 区间
    # 公式: lr = min + (max - min) * factor
    current_lr = min_lr + (base_lr - min_lr) * decay_factor

    return current_lr


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, is_global_test_client=False, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_rounds = args.global_rounds
        self.local_learning_rate_end = args.local_learning_rate_end
        self.lr_decay = args.lr_decay
        self.lr_decay_gamma = args.lr_decay_gamma  # exp模式的
        self.lr_warmup_round = args.lr_warmup_round  # warmup round
        self.weight_decay = args.weight_decay
        self.local_epochs = args.local_epochs
        self.save_gpu = args.save_gpu
        self.grad_clip_max_norm = args.grad_clip_max_norm
        self.num_workers = args.num_workers
        self.train_dataiter = None
        self.seed= args.seed
        self.checkpoint = args.checkpoint
        self.eval_only = args.eval_only
        self.train_client_names = args.train_client_names


        # 数据集选项
        self.fold_id = args.fold_id
        self.model_str = args.model_str

        # 直接加载数据集?
        # self.train_dataloader = self.load_train_data()
        # self.test_dataloader = self.load_test_data()

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.optimizer = None
        self.optimizer_name = args.optimizer_name


        if isinstance(self.id, int):
            self.seed_generator = torch.Generator().manual_seed(self.id)
        else:
            print(f"Site name {self.id} is not a int, set default seed generator, do not use client {self.id} for training!")
            self.seed_generator = torch.Generator().manual_seed(0)

        if not self.save_gpu:
            self.model = self.model.to(self.device)
        # 设置测试选项和loss
        self.loss = None
        self.rs_test_metrics = dict()
        self.rs_train_metrics = dict()
        self.available_modality_names = []
        #
        self.all_modality_names = []
        if self.dataset == "tcga":
            self.loss = NLLSurvLoss(alpha=0)
            self.rs_test_metrics ={
                'c_index': C_Index().to(self.device),
                'loss': Scalar().to(self.device)
            }
            self.rs_val_metrics = {
                'c_index': C_Index().to(self.device),
                'loss': Scalar().to(self.device)
            }
            self.rs_train_metrics = {
                'loss': Scalar().to(self.device)
            }

    def reset_optimizer(self, rnd):
        # self.optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, self.model.parameters()),
        #                                   lr=self.learning_rate, weight_decay=self.weight_decay, betas=(0.9, 0.999), )
        if self.lr_decay is not None:
            lr_init = get_fl_lr(rnd, total_rounds=self.global_rounds,
                                           base_lr=self.learning_rate, min_lr=self.local_learning_rate_end,
                                           mode=self.lr_decay,warmup_rounds=self.lr_warmup_round)
            if rnd == 0:
                print(f"optimizer_name: {self.optimizer_name}, init_lr={self.learning_rate}, decay={self.lr_decay}, warmup_rounds={self.lr_warmup_round}"
                      f",end_lr={self.local_learning_rate_end}, wd={self.weight_decay}")
        else:
            lr_init = self.learning_rate
        # print("lr_init: {}".format(lr_init))
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=lr_init, weight_decay=self.weight_decay,
                                              betas=(0.9, 0.999), )
        elif self.optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                              lr=lr_init, weight_decay=self.weight_decay,
                                              betas=(0.9, 0.999),)
        elif self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                              lr=lr_init, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise NotImplementedError("Unsupported optimizer name: %s" % self.optimizer_name)

    def load_train_data(self, batch_size=None, num_workers=None):
        if batch_size == None:
            batch_size = self.batch_size
        if num_workers == None:
            num_workers = self.num_workers
        site_name = self.id
        if isinstance(self.id, int):
            site_name = self.train_client_names[self.id]
        train_data = read_client_data(dataset=self.dataset, split="train", site_name=site_name,
                                      batch_size=batch_size,
                                      model_name=self.model_str,
                                      num_workers=num_workers,
                                      seed=self.seed,
                                      fold_id=self.fold_id)
        return train_data

    def load_test_data(self, batch_size=None, num_workers=None):
        if batch_size == None:
            batch_size = self.batch_size
        if num_workers == None:
            num_workers = self.num_workers
        site_name = self.id
        if isinstance(self.id, int):
            site_name = self.train_client_names[self.id]
        test_data = read_client_data(dataset=self.dataset, split="test", site_name=site_name,
                                     batch_size=batch_size,
                                     model_name=self.model_str,
                                     num_workers=num_workers,
                                     seed=self.seed,
                                     fold_id=self.fold_id)
        return test_data

    def load_val_data(self, batch_size=None, num_workers=None):
        if batch_size == None:
            batch_size = self.batch_size
        if num_workers == None:
            num_workers = self.num_workers
        site_name = self.id
        if isinstance(self.id, int):
            site_name = self.train_client_names[self.id]
        val_data = read_client_data(dataset=self.dataset, split="val", site_name=site_name,
                                    batch_size=batch_size,
                                    model_name=self.model_str,
                                    num_workers=num_workers,
                                    seed=self.seed,
                                    fold_id=self.fold_id)
        return val_data


    def set_parameters(self, model: nn.Module):
        # ks = list(self.get_trainable_parameters(model, state_dict=state_dict, return_names=True).keys())
        # ss = list(self.get_trainable_parameters(self.model, state_dict=state_dict, return_names=True).keys())
        # i = 0
        for (new_name, new_param), (old_name, old_param) in zip(model.state_dict().items(), self.model.state_dict().items()):
            # old_param.data = new_param.data.clone()
            # print("src:",ks[i], new_param.shape, ", desc: ", ss[i],old_param.shape)
            old_param.data.copy_(new_param.data)
            # i += 1

    def clone_model(self, model, target, state_dict=False):
        for param, target_param in zip(self.get_trainable_parameters(model, state_dict=state_dict), self.get_trainable_parameters(target, state_dict=state_dict)):
            # target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()
            target_param.data.copy_(param.data)

    def update_parameters(self, model, new_params, state_dict=False):
        for param, new_param in zip(self.get_trainable_parameters(model, state_dict=state_dict), new_params):
            # param.data = new_param.data.clone()
            param.data.copy_(new_param.data)

    def forward_model(self, batch_dict, model=None, **kwargs):
        if model is None:
            model = self.model
        if isinstance(model, (AB_MIL, AB_MIL_with_LoRA, AB_MIL_with_LoRA2,
                              TRANS_MIL, WIKG_MIL, TransMIL2)):
            res = model(batch_dict['feature'], **kwargs)
            logits = res['logits']
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            loss = self.loss(hazards=hazards, S=S, Y=batch_dict['label'].long(), c=batch_dict['censorship'])
            result = {'loss': loss, 'logits': logits}
            return result
        elif isinstance(model, MiCo):
            res = model(data=batch_dict['feature'], **kwargs)
            logits = res['logits']
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            loss = self.loss(hazards=hazards, S=S, Y=batch_dict['label'].long(), c=batch_dict['censorship'])
            result = {'loss': loss, 'logits': logits}
            return result
        elif isinstance(model, (ABMILModel, ABMILModel2, WIKGMILModel,TransMILModel)):
            # MIL-labe格式的
            results_dict, log_dict = model(batch_dict['feature'],
                                           loss_fn=None,
                                           label=None,
                                           return_attention=True,
                                           return_slide_feats=True,
                                           **kwargs
                                           )
            logits = results_dict['logits']
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            loss = self.loss(hazards=hazards, S=S, Y=batch_dict['label'].long(), c=batch_dict['censorship'])
            base_res = log_dict
            base_res.update({'loss': loss, "logits": logits})
            return base_res
        else:
            raise NotImplementedError("Unsupported model type: %s" % type(model))

    def forward_base(self, batch_dict, model=None, **kwargs):
        if model is None:
            model = self.model

        if isinstance(model, (ABMILModel, ABMILModel2, WIKGMILModel,TransMILModel)):
            h, res = model.forward_features(h=batch_dict['feature'], **kwargs)
            res['feature_output'] = h
            return res
        else:
            if hasattr(model, "forward_feature"):
                res = model.forward_feature(batch_dict['feature'], **kwargs)
                return res
            raise NotImplementedError(f"{self.model.__class__.__name__} is not implemented for forwarding base feature.")

    def forward_head(self, batch_dict, feature_dict, model=None, **kwargs):
        if model is None:
            model = self.model
        if isinstance(model, (ABMILModel, ABMILModel2, WIKGMILModel,TransMILModel)):
            logits = model.forward_head(feature_dict['feature_output'], **kwargs)
            # cls_loss = MIL.compute_loss(loss_fn, logits, label)
            # results_dict = {'logits': logits, 'loss': cls_loss}
            # log_dict['loss'] = cls_loss.item() if cls_loss is not None else -1
            # if return_slide_feats:
            #     log_dict['slide_feats'] = wsi_feats
            # return results_dict, log_dict

            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            loss = self.loss(hazards=hazards, S=S, Y=batch_dict['label'].long(), c=batch_dict['censorship'])
            result = {'loss': loss, 'logits': logits}
            return result
        else:
            if hasattr(model, "forward_head"):
                res = model.forward_head(feature_dict, **kwargs)
                logits = res['logits']
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                loss = self.loss(hazards=hazards, S=S, Y=batch_dict['label'].long(), c=batch_dict['censorship'])
                result = {'loss': loss, 'logits': logits}
                return result
            raise NotImplementedError(
                f"{self.model.__class__.__name__} is not implemented for forwarding head feature.")

    def update_metrics(self, output, input_dict, data_split: Literal['train', 'val', 'test'], **kwargs):
        if data_split == "train":
            metric_objs = self.rs_train_metrics
        elif data_split == "val":
            metric_objs = self.rs_val_metrics
        else:
            metric_objs = self.rs_test_metrics
        for metric_key, obj in metric_objs.items():
            if 'c_index' in metric_key:
                batch_v = obj.update(logits=output, censorship=input_dict['censorship'], event_time=input_dict['event_time'])
            elif 'loss' in metric_key:
                batch_v = obj.update(kwargs['loss'].item())
            else:
                raise NotImplementedError

    def describe_metrics(self, data_split: Literal['train', 'val', 'test']) -> Dict[str, float]:
        if data_split == "train":
            metric_objs = self.rs_train_metrics
        elif data_split == "val":
            metric_objs = self.rs_val_metrics
        else:
            metric_objs = self.rs_test_metrics

        # 统计信息
        metrics = dict()
        for metric_key, obj in metric_objs.items():
            metrics[metric_key] = obj.compute().item()
            obj.reset()
        return metrics

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
                output_dict = self.forward_model(batch_dict)
                loss = output_dict['loss']
                self.update_metrics(output_dict['logits'], batch_dict, data_split=data_split, is_train=False, loss=loss)

                test_num += batch_dict['feature'].shape[0]  # dim0是具体的sample数量. dim1是patch级别的数量

        if self.save_gpu:
            self.model.cpu()
            # self.generator.cpu()
        return test_num, self.describe_metrics(data_split=data_split)


    def train_metrics(self) -> Tuple[float, int, Dict[str, float]]:
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        if self.save_gpu:
            self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for batch_dict in trainloader:
                batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
                output = self.forward_model(batch_dict)
                loss = self.loss(output, batch_dict['label'])
                self.update_metrics(output, batch_dict, is_train=True, loss=loss)


                train_num += batch_dict["label"].shape[0]
                losses += loss.item() * batch_dict["label"].shape[0]

        if self.save_gpu:
            self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num, self.describe_metrics(is_train=True)

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

if __name__ == '__main__':
    base_lr = 0.0001
    lrs = []
    xs =[]
    for rnd in range(0, 20):
        lr = get_fl_lr(rnd, total_rounds=20,
                                           base_lr=base_lr, min_lr=1e-6,
                                           mode='cosine',warmup_rounds=5)
        lrs.append(lr)
        xs.append(rnd)
    import matplotlib.pyplot as plt
    plt.plot(xs, lrs)
    plt.show()