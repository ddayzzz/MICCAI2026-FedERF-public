import time
from argparse import ArgumentParser
from collections import defaultdict

import math
import pandas as pd
import torch
import numpy as np
import random
import os

from PIL.ImageFile import raise_oserror
from sksurv.metrics import concordance_index_censored
from sklearn.utils import resample
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch.nn as nn
from tqdm import tqdm

from data.tcga_dataset import TCGADataset
from flcore.trainmodel.AB_MIL import AB_MIL, AB_MIL_with_LoRA, AB_MIL_with_LoRA2

from flcore.trainmodel.WIKG_MIL import WIKG_MIL
from flcore.trainmodel.hypernet import Prototype2LoRA_Hypernet, FunctionalLoRALinear, MoEFunctionalLoRA
from flcore.trainmodel.hypernet2 import ViTHyper
from flcore.trainmodel.lora_modules.lora_layers import FunctionalLinear
from flcore.trainmodel.lora_modules.lora_utils import mark_only_lora_as_trainable, mark_only_weight_as_trainable
from flcore.trainmodel.mico import MiCo
from flcore.trainmodel.mil_models.abmil import ABMILGatedBaseConfig, ABMILModel
from flcore.trainmodel.mil_models.transmil import TransMILConfig, TransMILModel
from flcore.trainmodel.mil_models.wikg_mil import WIKGMILModel, WIKGConfig
from flcore.trainmodel.my_metrics import C_Index
from flcore.trainmodel.route_modules import ProtoGuidedLoRA, MoEConvNDLoRA
from flcore.trainmodel.trans_mil.trans_mil import TRANS_MIL
from flcore.trainmodel.trans_mil.trans_mil2 import TransMIL2
from flcore.utils.read_yaml import read_yaml
from flcore.utils.survial_loss import NLLSurvLoss

from lifelines.statistics import logrank_test
import numpy as np

from torch.optim.lr_scheduler import LambdaLR


def create_scheduler(optimizer, base_lr, num_epochs, warmup_epochs=0, mode='cosine', min_lr=1e-6):
    """
    创建一个支持 Warmup 和多种衰减模式的学习率调度器。

    参数:
        optimizer: 优化器对象
        num_epochs (int): 总训练轮数
        warmup_epochs (int): 预热轮数 (建议设置为 num_epochs 的 5%-10%)
        mode (str): 'cosine', 'linear', 或 'step'
        min_lr (float): 训练结束时的最小学习率 (防止 LR 变为 0)

    返回:
        scheduler: PyTorch 的 LambdaLR 对象
    """

    # 获取优化器设定的初始学习率 (base_lr)
    # base_lr = optimizer.param_groups[0]['lr']

    def lr_lambda(current_epoch):
        # -------------------------
        # 1. Warmup 阶段
        # -------------------------
        if current_epoch < warmup_epochs:
            # 线性增长: 0 -> 1 (相对于 base_lr 的比例)
            return float(current_epoch + 1) / float(max(1, warmup_epochs))

        # -------------------------
        # 2. Decay (衰减) 阶段
        # -------------------------
        # 进度 progress: 0.0 (warmup结束) -> 1.0 (训练结束)
        progress = float(current_epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))

        # 计算衰减因子 (0到1之间)
        if mode == 'cosine':
            # 余弦衰减: 1 -> 0
            decay_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif mode == 'linear':
            # 线性衰减: 1 -> 0
            decay_factor = max(0.0, 1.0 - progress)
        elif mode == 'step':
            # 阶梯衰减: 每过一半时间减半 (简单示例)
            decay_factor = 0.5 ** int(progress * 3)  # 这里的3控制阶梯数
        else:
            decay_factor = 1.0  # 不衰减

        # -------------------------
        # 3. 结合 Min_LR
        # -------------------------
        # 公式: lr = min_lr + (base_lr - min_lr) * decay_factor
        # 因为 LambdaLR 返回的是乘法因子，所以我们要计算这个因子：
        # factor = (min_lr + (base_lr - min_lr) * decay_factor) / base_lr

        target_lr = min_lr + (base_lr - min_lr) * decay_factor
        return target_lr / base_lr

    return LambdaLR(optimizer, lr_lambda)

def cox_log_rank(hazards, labels, survtime_all):
    hazardsdata = hazards.reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata < median] = 1
    survtime_all = survtime_all.reshape(-1)
    idx = hazards_dichotomize == 0
    labels = labels
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return (pvalue_pred)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloader(dataset_name, split, fold_id, n_shot=None, seed=42, enable_roupkt=False, search_experts_dir=None, loaded_expert_names=None):
    cfg = read_yaml(f'config/data/{dataset_name}_survival.yaml')
    if split == "train":
        if enable_roupkt:
            Mydata = TCGADataset(dataset_cfg=cfg.Data, split='train', fold_id=fold_id,
                                 load_experts_names=loaded_expert_names, search_experts_dir=search_experts_dir,
                                 n_shot=n_shot, seed=seed)
        else:
            Mydata = TCGADataset(dataset_cfg=cfg.Data, split='train', fold_id=fold_id,n_shot=n_shot, seed=seed)
        dataloader = DataLoader(Mydata, num_workers=4, shuffle=True, batch_size=1)
    elif split == "val" or split == "test":
        if enable_roupkt:
            Mydata = TCGADataset(dataset_cfg=cfg.Data, split=split, fold_id=fold_id, load_experts_names=loaded_expert_names, search_experts_dir=search_experts_dir)
        else:
            Mydata = TCGADataset(dataset_cfg=cfg.Data, split=split, fold_id=fold_id)
        dataloader = DataLoader(Mydata, num_workers=4, shuffle=False, batch_size=1)
    return dataloader

def train(model, train_dataloader, optimizer, device, epoch):
    model.train()
    model.to(device)

    criterion = NLLSurvLoss(alpha=0)
    sum_loss = 0
    bi = 0
    for data_dict in train_dataloader:
        feature, label, event_time, censorship = data_dict['feature'], data_dict['label'], data_dict['event_time'], data_dict['censorship']
        feature = feature.to(device)
        label = label.to(device).long()
        event_time = event_time.to(device)
        censorship = censorship.to(device)

        if isinstance(model, MiCo):
            return_dict = model(data=feature)
        elif isinstance(model, (ABMILModel, WIKGMILModel, TransMILModel)):
            add_input = dict()
            if enable_roupkt:
                add_input['X_tfl'] = data_dict['expert_features'].to(device)
            return_dict, log_dict = model(feature,
                                          loss_fn=None,
                                          label=None,
                                          return_attention=True,
                                          return_slide_feats=True,
                                          **add_input)
        else:
            return_dict = model(feature)
        logits = return_dict['logits']
        # Y_hat = torch.argmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        loss = criterion(hazards=hazards, S=S, Y=label, c=censorship)

        if enable_roupkt:
            router_z_loss = return_dict['router_z_loss']
            balance_loss = return_dict['balance_loss']
            loss_router_z_weight = 0.01
            loss_balance_weight = 0.01
            loss += (router_z_loss * loss_router_z_weight + balance_loss * loss_balance_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bi += 1
        # print(f"batch {bi}, shape: {feature.size()} finished!", flush=True)
        sum_loss += loss.item()

    print(f"Epoch[{epoch:<5}], train loss: {sum_loss / len(train_dataloader):.4f}")


def train_lora(model, train_dataloader, optimizer, device, epoch, router_optimizer=None):
    model.train()
    model.to(device)
    criterion = NLLSurvLoss(alpha=0)
    sum_loss = 0
    bi = 0
    for data_dict in train_dataloader:

        feature, label, event_time, censorship = data_dict['feature'], data_dict['label'], data_dict['event_time'], data_dict['censorship']
        feature = feature.to(device)
        label = label.to(device).long()
        event_time = event_time.to(device)
        censorship = censorship.to(device)

        if isinstance(model, MiCo):
            return_dict = model(data=feature)
        elif isinstance(model, (ABMILModel, WIKGMILModel, TransMILModel)):
            return_dict, log_dict = model(feature,
                                          loss_fn=None,
                                          label=None,
                                          return_attention=True,
                                          return_slide_feats=True,
                                          )
        else:
            return_dict = model(feature)
        logits = return_dict['logits']
        # Y_hat = torch.argmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        loss = criterion(hazards=hazards, S=S, Y=label, c=censorship)

        optimizer.zero_grad()
        if router_optimizer is not None:
            router_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if router_optimizer is not None:
            router_optimizer.step()
        bi += 1
        # print(f"batch {bi}, shape: {feature.size()} finished!", flush=True)
        sum_loss += loss.item()

    print(f"Epoch[{epoch:<5}], train loss: {sum_loss / len(train_dataloader):.4f}")


@torch.no_grad()
def evaluate(model, dataloader, device, epoch):
    model.eval()
    criterion = NLLSurvLoss(alpha=0)
    save_metrics = defaultdict(list)

    for data_dict in dataloader:
        feature, label, event_time, censorship = data_dict['feature'], data_dict['label'], data_dict['event_time'], \
        data_dict['censorship']
        feature = feature.to(device)
        label = label.to(device).long()
        event_time = event_time.to(device)
        censorship = censorship.to(device)

        if isinstance(model, MiCo):
            return_dict = model(data=feature)
        elif isinstance(model, (ABMILModel, WIKGMILModel, TransMILModel)):
            return_dict, log_dict = model(feature,
                                           loss_fn=None,
                                           label=None,
                                           return_attention=True,
                                           return_slide_feats=True
                                           )
        else:
            return_dict = model(feature)
        logits = return_dict['logits']
        # Y_hat = torch.argmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S, dim=1).cpu().item()
        loss = criterion(hazards=hazards, S=S, Y=label, c=censorship)
        # return {'loss': loss.item(), 'risk': risk, 'censorship': c.item(), 'event_time': event_time.item()}
        save_metrics['loss'].append(loss.item())
        save_metrics['risk'].append(risk)
        save_metrics['censorship'].append(censorship.item())
        save_metrics['event_time'].append(event_time.item())
    all_val_loss = np.stack(save_metrics['loss'])
    all_risk_scores = np.stack(save_metrics['risk'])
    all_censorships = np.stack(save_metrics['censorship'])
    all_event_times = np.stack(save_metrics['event_time'])

    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores,
                                         tied_tol=1e-08)[0]
    pvalue_pred = cox_log_rank(all_risk_scores, (1 - all_censorships), all_event_times)

    # self.log('val_loss', np.mean(all_val_loss), prog_bar=True, on_epoch=True, logger=True)
    # self.log('c_index', c_index, prog_bar=True, on_epoch=True, logger=True)
    # self.log('p_value', pvalue_pred, prog_bar=True, on_epoch=True, logger=True)
    epoch_loss = np.mean(all_val_loss)
    return epoch_loss, c_index, pvalue_pred

@torch.no_grad()
def testting(model, dataloader, device, epoch):
    model.eval()
    save_metrics = defaultdict(list)

    for data_dict in dataloader:
        feature, label, event_time, censorship = data_dict['feature'], data_dict['label'], data_dict['event_time'], \
        data_dict['censorship']
        feature = feature.to(device)
        label = label.to(device).long()
        event_time = event_time.to(device)
        censorship = censorship.to(device)

        # return_dict = model(feature)
        if isinstance(model, MiCo):
            return_dict = model(data=feature)
        elif isinstance(model, (ABMILModel, WIKGMILModel, TransMILModel)):
            add_input = dict()
            if enable_roupkt:
                add_input['X_tfl'] = data_dict['expert_features'].to(device)
            return_dict, log_dict = model(feature,
                                          loss_fn=None,
                                          label=None,
                                          return_attention=True,
                                          return_slide_feats=True,
                                          **add_input
                                          )
        else:
            return_dict = model(feature)
        logits = return_dict['logits']
        # Y_hat = torch.argmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S, dim=1).cpu().item()
        # return {'loss': loss.item(), 'risk': risk, 'censorship': c.item(), 'event_time': event_time.item()}
        save_metrics['risk'].append(risk)
        save_metrics['censorship'].append(censorship.item())
        save_metrics['event_time'].append(event_time.item())
    all_risk_scores = np.stack(save_metrics['risk'])
    all_censorships = np.stack(save_metrics['censorship'])
    all_event_times = np.stack(save_metrics['event_time'])

    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores,
                                         tied_tol=1e-08)[0]
    pvalue_pred = cox_log_rank(all_risk_scores, (1 - all_censorships), all_event_times)
    # print(f'Test C-Index: {c_index:.4f}, P-Value: {pvalue_pred:.4f}')

    # Bootstrap
    n_boots = 1000
    boot_c_indices = []
    for i in tqdm(range(n_boots), desc="Bootstrapping C-Index"):
        boot_ids = resample(np.arange(len(all_risk_scores)), replace=True)
        try:
            c_index_boot = concordance_index_censored(
                (1 - all_censorships[boot_ids]).astype(bool),
                all_event_times[boot_ids],
                all_risk_scores[boot_ids],
                tied_tol=1e-08
            )[0]
            boot_c_indices.append(c_index_boot)
        except ZeroDivisionError:
            continue

    # 计算置信区间
    ci_lower, ci_upper = np.percentile(boot_c_indices, [2.5, 97.5])
    # print(f"95% CI for C-Index: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # 保存结果
    results_dict = {
        'c_index': c_index,
        'c_index_low': ci_lower,
        'c_index_high': ci_upper,
        'p_value': pvalue_pred
    }

    return c_index, ci_lower, ci_upper, pvalue_pred, results_dict

def load_checkpoint(checkpoint_path, model, use_pretrained_classifier=False, only_finetune_classifier=False):
    ignore_dict = {
        ABMILModel: ['model.classifier'],
        WIKGMILModel: ['model.classifier'],
        TransMILModel: ['model.classifier'],
        MiCo: ['classifier'],
    }
    if checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        print("Load checkpoint from {}".format(args.checkpoint))
        strict_load = True


        if not use_pretrained_classifier:
            new_st = dict()

            ignore_names = ignore_dict[model.__class__]
            for k, v in checkpoint.items():
                if any([k.startswith(iname) for iname in ignore_names]):
                    continue
                new_st[k] = v
            checkpoint = new_st
            strict_load = False
        misk, une_k = model.load_state_dict(checkpoint, strict=strict_load)
        print("Loading pretrained(use_pretrained_classifier={}). Missing keys: {}, unexpcted keys: {}".format(use_pretrained_classifier,misk, une_k))
    else:
        print("No checkpoint found.")
    if only_finetune_classifier:
        forzened_pnames= []
        ignore_names = ignore_dict[model.__class__]
        for pname, pvar in model.named_parameters():
            if not (any([pname.startswith(iname) for iname in ignore_names])):
                forzened_pnames.append(pname)
                pvar.requires_grad = False
        print("Frozen parameter names: {}".format(forzened_pnames))


@torch.no_grad()
def load_checkpoint_lora(checkpoint_path, model, use_pretrained_classifier=False, only_finetune_classifier=False,
                         fix_expert=False, no_expert_initial_expert_num=None, no_expert=False,
                         average_route_scores=False):
    ignore_dict = {
        ABMILModel: ['model.classifier'],
        WIKGMILModel: ['model.classifier'],
        TransMILModel: ['model.classifier'],
        MiCo: ['classifier'],
    }
    if checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        print("Load checkpoint from {}".format(args.checkpoint))
        # 加载通用的特征
        backbone_model_state_dict = checkpoint['global']
        # backbone_model_state_dict = checkpoint
        if not use_pretrained_classifier:
            new_st = dict()

            ignore_names = ignore_dict[model.__class__]
            for k, v in backbone_model_state_dict.items():
                if any([k.startswith(iname) for iname in ignore_names]):
                    continue
                new_st[k] = v
            backbone_model_state_dict = new_st
        misk, une_k = model.load_state_dict(backbone_model_state_dict, strict=False)
        print("Loading pretrained(use_pretrained_classifier={}). Missing keys: {}, unexpcted keys: {}".format(use_pretrained_classifier,misk, une_k))
        # 加载路由的各种特征
        lora_state_dicts = []
        client_names = []
        if no_expert:
            print(f"Disable expert initialization, random initialize {no_expert_initial_expert_num} experts.")
            sub_state_dict = checkpoint["client_models"][list(checkpoint["client_models"].keys())[0]]  # 模拟的第一个!
            st_a = {k: v for k, v in sub_state_dict.items() if 'lora_A' in k}
            st_b = {k: v for k, v in sub_state_dict.items() if 'lora_B' in k}
            for ci in range(no_expert_initial_expert_num):
                new_init_st = dict()
                for stk, stv in st_a.items():
                    lora_a = torch.empty_like(stv)
                    lora_b = torch.empty_like(st_b[stk.replace('lora_A', 'lora_B')])
                    # 初始化!
                    nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
                    nn.init.zeros_(lora_b)
                    new_init_st[stk] = lora_a
                    new_init_st[stk.replace('lora_A', 'lora_B')] = lora_b
                lora_state_dicts.append(new_init_st)
                client_names.append("random_expert{}".format(ci))
        else:
            for client_name, sub_state_dict in checkpoint["client_models"].items():
                client_names.append(client_name)
                # 生成具体的lora参数
                lora_state_dicts.append({k: v for k, v in sub_state_dict.items() if 'lora_' in k})
        # for ci in range(5):
        #     client_names.append(f"{ci}")
        #     lora_state_dicts.append({k: v for k, v in checkpoint.items() if 'lora_' in k})
        # lora_state_dicts.append({k: v for k, v in backbone_model_state_dict.items() if 'lora_' in k})
        for module in model.modules():
            if isinstance(module, (ProtoGuidedLoRA, MoEConvNDLoRA)):
                module._mix_trajectories(lora_state_dicts, frozen_experts=fix_expert,
                                         route_act="softmax",
                                         disable_scaling=False,
                                         average_route=average_route_scores)

    else:
        print("No checkpoint found.")
    if only_finetune_classifier:
        route_map = {
            ABMILModel: ['_routing_fn', 'lora_'],
            WIKGMILModel: ['_routing_fn', 'lora_'],
            TransMILModel: ['_routing_fn', 'lora_'],
            MiCo: ['_routing_fn', 'lora_'],
        }
        forzened_pnames= []
        ignore_names = ignore_dict[model.__class__]
        for pname, pvar in model.named_parameters():
            is_not_clf = not (any([pname.startswith(iname) for iname in ignore_names]))
            is_not_route = not (any([iname in pname for iname in route_map[model.__class__]]))
            if is_not_clf and is_not_route:
                forzened_pnames.append(pname)
                pvar.requires_grad = False
        print("Frozen parameter names: {}".format(forzened_pnames))


def create_optimizer_scheduler(model: nn.Module, optimizer_name, weight_decay, backbone_lr, classifier_lr, lora_lr, lora_wd):
    classifier_names = {
        ABMILModel: ['model.classifier'],
        WIKGMILModel: ['model.classifier'],
        TransMILModel: ['model.classifier'],
        MiCo: ['classifier'],

    }
    router_optimizer = None
    if isinstance(model, (ABMILModel, WIKGMILModel, TransMILModel, MiCo)):
        clf_param_groups = []
        backbone_param_groups = []
        if model_str.endswith("_lora"):
            trainable_param_all_names = ['_routing_fn', 'lora_']
        elif model_str.endswith("roupkt"):
            trainable_param_all_names = ['experts.', 'router.']
        else:
            trainable_param_all_names = ['_routing_fn', 'lora_']
        trainable_params = []
        exited_trainable_param_names = []

        clf_names, backbone_names = [], []
        for name, param in model.named_parameters():
            clf_existed = any([name.startswith(n) for n in classifier_names[model.__class__]])
            is_trained = any([n in name for n in trainable_param_all_names])
            if clf_existed and not is_trained:
                clf_param_groups.append(param)
                clf_names.append(name)
        for name, param in model.named_parameters():
            not_clf_existed = not (any([name.startswith(n) for n in classifier_names[model.__class__]]))
            is_trained = any([n in name for n in trainable_param_all_names])
            if not_clf_existed and not is_trained:
                backbone_param_groups.append(param)
                backbone_names.append(name)

        for name, param in model.named_parameters():
            is_trained = any([n in name for n in trainable_param_all_names])
            if is_trained:
                trainable_params.append(param)
                exited_trainable_param_names.append(name)

        pg = [
            {'params': clf_param_groups, 'lr': classifier_lr, "weight_decay": weight_decay},
            {'params': backbone_param_groups, 'lr': backbone_lr, "weight_decay": weight_decay},

        ]

        if len(exited_trainable_param_names) > 0:
            pg.append({'params': trainable_params, 'lr': lora_lr, "weight_decay": lora_wd})
            print("lora, router params: {}, clf lr: {}, wd: {}".format(exited_trainable_param_names, lora_lr,lora_wd  ))
            # router_optimizer = torch.optim.AdamW(trainable_params, lr=lora_lr, weight_decay=lora_wd)
            # router_optimizer = None
    else:
        clf_param_groups = []
        backbone_param_groups = []

        clf_names, backbone_names = [], []
        for name, param in model.named_parameters():
            clf_existed = any([name.startswith(n) for n in classifier_names[model.__class__]])
            if clf_existed:
                clf_param_groups.append(param)
                clf_names.append(name)
        for name, param in model.named_parameters():
            not_clf_existed = not (any([name.startswith(n) for n in classifier_names[model.__class__]]))
            if not_clf_existed:
                backbone_param_groups.append(param)
                backbone_names.append(name)

        pg = [
            {'params': clf_param_groups, 'lr': classifier_lr, "weight_decay": weight_decay},
            {'params': backbone_param_groups, 'lr': backbone_lr, "weight_decay": weight_decay},
        ]
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(pg, betas=(0.9, 0.999), )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(pg, betas=(0.9, 0.999), )
    print("Backbone params: {}, lr: {}, wd: {}".format(backbone_names, backbone_lr, weight_decay))
    print("classifier params: {}, lr: {}, wd:{}".format(clf_names, classifier_lr, weight_decay))

    scheduler = None
    if lr_decay is not None:
        print("lr decay, mode={}, end_lr: {}, warmup[no used]: {}".format(lr_decay, end_lr, lr_warmup))
        if lr_decay == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=end_lr)
        else:
            raise NotImplementedError
    return optimizer, scheduler, router_optimizer


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--end_lr", type=float, default=1e-6)
    parser.add_argument("--lr_decay", type=str, default='cosine', choices=['linear', 'cosine', 'constant'])
    parser.add_argument("--lr_warmup", type=int, default=2)
    parser.add_argument("--feature_lr", type=float, default=1e-4)
    parser.add_argument("--lora_lr", type=float, default=1e-3)
    parser.add_argument("--lora_wd", type=float, default=1e-2)
    parser.add_argument("--optimizer", type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--train_data_frac", type=int, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--base_seed", type=int, default=2025)
    parser.add_argument("--fold_id", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--num_fold", type=int, default=4)
    parser.add_argument("--load_pretrain_classifier", default=False, action='store_true')
    parser.add_argument("--only_finetune_classifier", default=False, action='store_true')
    parser.add_argument("--fix_expert", default=False, action='store_true')
    parser.add_argument("--no_expert", default=False, action='store_true')
    parser.add_argument("--average_route_scores", default=False, action='store_true')
    parser.add_argument("--no_expert_initial_expert_num", default=None, type=int)
    parser.add_argument("--model", type=str, default='ab_mil', )
    parser.add_argument("--n_shot", type=int, default=None, )


    args = parser.parse_args()
    print("=" * 50)
    for arg in vars(args):
        print(arg, '=', getattr(args, arg))
    print("=" * 50)

    seed = args.base_seed # + args.run_id
    setup_seed(seed)
    max_epochs = args.max_epochs
    learning_rate = args.lr
    weight_decay = args.weight_decay
    dataset = args.dataset
    batch_size = args.batch_size
    fold_id = args.fold_id
    model_str = args.model
    device = torch.device(f"cuda:{args.gpu}")
    lr_decay = args.lr_decay
    lr_warmup = args.lr_warmup
    end_lr = args.end_lr
    n_shot = args.n_shot
    # search_experts_dir = f"/xxx/projects2/fedmed2026/roupkt_expert_outputs/ab_mil_tiny_fold{fold_id}/transfer_to_{dataset}"
    search_experts_dir = f"./roupkt_expert_outputs/ab_mil_tiny_fold{fold_id}/transfer_to_{dataset}"
    all_experts_names = ['LIHC','SARC', 'STAD', 'SKCM', 'UCEC']
    load_experts_names = [cc for cc in all_experts_names if cc != dataset]
    enable_roupkt = model_str.endswith("roupkt")

    if n_shot is not None:
        save_dir = "finetune_results/finetune_{}_{}_{}shot/exp_{}".format(dataset, model_str, n_shot, args.exp_name)
    else:
        save_dir = "finetune_results/finetune_{}_{}/exp_{}".format(dataset, model_str, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    train_dataloader = get_dataloader(dataset, split='train', fold_id=fold_id, n_shot=n_shot, seed=seed + fold_id, enable_roupkt=enable_roupkt,
                                      loaded_expert_names=load_experts_names,
                                      search_experts_dir=search_experts_dir)
    # val_dataloader = get_dataloader(dataset, split='val')
    test_dataloader = get_dataloader(dataset, split='test', fold_id=fold_id, enable_roupkt=enable_roupkt,
                                     loaded_expert_names=load_experts_names,
                                     search_experts_dir=search_experts_dir)
    print(f"Num of train size: {len(train_dataloader.dataset)}, Num of test size: {len(test_dataloader.dataset)}")

    if model_str == "ab_mil":  # convex
        # model = AB_MIL(num_classes=args.num_classes)
        config = ABMILGatedBaseConfig(in_dim=512, embed_dim=384, attn_dim=256, num_classes=args.num_classes)
        model = ABMILModel(config)
    elif model_str == "ab_mil_tiny":
        config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=64, num_classes=args.num_classes)  # 0209 修改了attention 的dim为64
        model = ABMILModel(config)
    elif model_str == "ab_mil_tiny_roupkt":
        config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=64, num_classes=args.num_classes,
                                      enable_roupkt=True,
                                      expert_network="MLP")  # 0209 修改了attention 的dim为64
        model = ABMILModel(config)
    # elif model_str == "ab_mil_tiny_rank16_alpha4_fc4_lora":
    #     config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes,
    #                                   embedding_rank=16, embedding_alpha=4,
    #                                   attention_rank=16, attention_alpha=4,
    #                                   classifier_rank=None, classifier_alpha=None,
    #                                   final_attention_lora_multi=1, add_route=True
    #                                   )
    #     model = ABMILModel(config)
    # elif model_str == "ab_mil_tiny_rankMidFeat_alphax2_fcNone_lora":
    #     config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes,
    #                                   embedding_rank=64, embedding_alpha=128,
    #                                   attention_rank=16, attention_alpha=32,
    #                                   classifier_rank=None, classifier_alpha=None,
    #                                   final_attention_lora_multi=1, add_route=True
    #                                   )
    #     model = ABMILModel(config)
    # elif model_str == "ab_mil_tiny_rankFullFeat_alphax2_fcNone_lora":
    #     config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes,
    #                                   embedding_rank=128, embedding_alpha=256,
    #                                   attention_rank=32, attention_alpha=64,
    #                                   classifier_rank=None, classifier_alpha=None,
    #                                   final_attention_lora_multi=1,add_route=True
    #                                   )
    #     model = ABMILModel(config)
    elif model_str == "ab_mil_tiny_rankMidFeat_alphax0.25_lora":
        config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=64, num_classes=args.num_classes,
                                      embedding_rank=64, embedding_alpha=16,
                                      attention_rank=16, attention_alpha=4,
                                      classifier_rank=2, classifier_alpha=0.5,
                                      final_attention_lora_multi=1, add_route=True
                                      )
        model = ABMILModel(config)
    elif model_str == "ab_mil_tiny_rank0.25Feat_alphax0.25_lora":
        # 意思是rank为最小维度的1/4
        config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=64, num_classes=args.num_classes,
                                      embedding_rank=32, embedding_alpha=8,
                                      attention_rank=8, attention_alpha=2,
                                      classifier_rank=1, classifier_alpha=0.25,
                                      final_attention_lora_multi=1,
                                      )
        model = ABMILModel(config)
    elif model_str == "ab_mil_tiny_rank0.75Feat_alphax0.25_lora":
        # 意思是rank为最小维度的3/4
        config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=64, num_classes=args.num_classes,
                                      embedding_rank=96, embedding_alpha=24,
                                      attention_rank=24, attention_alpha=6,
                                      classifier_rank=3, classifier_alpha=0.75,
                                      final_attention_lora_multi=1,
                                      )
        model = ABMILModel(config)
    elif model_str == "ab_mil_tiny_rank1.0Feat_alphax0.25_lora":
        config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=64, num_classes=args.num_classes,
                                      embedding_rank=128, embedding_alpha=32,
                                      attention_rank=32, attention_alpha=8,
                                      classifier_rank=4, classifier_alpha=1,
                                      final_attention_lora_multi=1,
                                      )
        model = ABMILModel(config)
    elif model_str == "ab_mil_tiny_rankMidFeat_alphax1.0_lora":
        config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=64, num_classes=args.num_classes,
                                      embedding_rank=64, embedding_alpha=64,
                                      attention_rank=16, attention_alpha=16,
                                      classifier_rank=2, classifier_alpha=2,
                                      final_attention_lora_multi=1,
                                      )
        model = ABMILModel(config)
    elif model_str == "ab_mil_tiny_rankMidFeat_alphax0.5_lora":
        config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=64, num_classes=args.num_classes,
                                      embedding_rank=64, embedding_alpha=32,
                                      attention_rank=16, attention_alpha=8,
                                      classifier_rank=2, classifier_alpha=1,
                                      final_attention_lora_multi=1,
                                      )
        model = ABMILModel(config)
    elif model_str == "ab_mil_tiny_rankMidFeat_alphax2.0_lora":
        config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=64, num_classes=args.num_classes,
                                      embedding_rank=64, embedding_alpha=128,
                                      attention_rank=16, attention_alpha=32,
                                      classifier_rank=2, classifier_alpha=4,
                                      final_attention_lora_multi=1,
                                      )
        model = ABMILModel(config)
    elif model_str == "ab_mil_lora":
        # model = AB_MIL_with_LoRA(num_classes=args.num_classes, add_router=True)
        # model = AB_MIL_with_LoRA2(num_classes=args.num_classes, add_router=True)
        # config = ABMILGatedBaseConfig(in_dim=512, embed_dim=384, attn_dim=256, num_classes=args.num_classes, add_route=True, lora_rank=8, lora_alpha=16)  # 0130
        config = ABMILGatedBaseConfig(in_dim=512, embed_dim=384, attn_dim=256, num_classes=args.num_classes, add_route=True, lora_rank=16, lora_alpha=8)  # 0202
        model = ABMILModel(config)
    elif model_str == "ab_mil_0204_lora":
        #
        config = ABMILGatedBaseConfig(in_dim=512, embed_dim=384, attn_dim=256, num_classes=args.num_classes,
                                      embedding_rank=16, embedding_alpha=32,
                                      attention_rank=16, attention_alpha=32,
                                      classifier_rank=16, classifier_alpha=32,
                                      final_attention_lora_multi=1,
                                      add_route=True)
        model = ABMILModel(config)
    elif model_str == "ab_mil_0205_rankMidMin_scale0.25_lora":
        config = ABMILGatedBaseConfig(in_dim=512, embed_dim=384, attn_dim=256, num_classes=args.num_classes,
                                      embedding_rank=192, embedding_alpha=48,
                                      attention_rank=32, attention_alpha=8,
                                      classifier_rank=2, classifier_alpha=0.5,
                                      final_attention_lora_multi=1, add_route=True)
        model = ABMILModel(config)
    elif model_str == "ab_mil_0205_rank4_alpha2_lora":
        config = ABMILGatedBaseConfig(in_dim=512, embed_dim=384, attn_dim=256, num_classes=args.num_classes,
                                      embedding_rank=4, embedding_alpha=2,
                                      attention_rank=4, attention_alpha=2,
                                      classifier_rank=4, classifier_alpha=2,
                                      final_attention_lora_multi=1,add_route=True )
        model = ABMILModel(config)
    elif model_str == "trans_mil":
        model = TRANS_MIL(num_classes=args.num_classes, dropout=0.1, act=nn.ReLU(), in_dim=512)
    elif model_str == "trans_mil2":
        # args.model = TransMIL2(num_classes=4, dropout=0.25, num_heads=4, num_attention_layers=2, embed_dim=256, in_dim=512, num_fc_layers=1)
        config = TransMILConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes,
                                num_heads=4, num_fc_layers=1, num_attention_layers=2
                                )
        model = TransMILModel(config)
    elif model_str == "trans_mil2_roupkt":
        # args.model = TransMIL2(num_classes=4, dropout=0.25, num_heads=4, num_attention_layers=2, embed_dim=256, in_dim=512, num_fc_layers=1)
        config = TransMILConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes,
                                num_heads=4, num_fc_layers=1, num_attention_layers=2,
                                enable_roupkt=True,
                                expert_network="MLP"
                                )
        model = TransMILModel(config)
    elif model_str == "trans_mil2_rankMidFeat_alphax0.25_lora":
        # args.model = TransMIL2(num_classes=4, dropout=0.25, num_heads=4, num_attention_layers=2, embed_dim=256, in_dim=512, num_fc_layers=1)
        config = TransMILConfig(in_dim=512, embed_dim=128, num_classes=args.num_classes,
                                num_heads=4, num_fc_layers=1, num_attention_layers=2,
                                patch_rank=64, patch_alpha=16,
                                ppeg_proj_rank=0.5, ppeg_proj_alpha=1,
                                classifier_rank=2, classifier_alpha=0.5,
                                add_route=True
                                )
        model = TransMILModel(config)
    elif model_str == "wikg_mil":
        config = WIKGConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes, topk=4,
                            agg_type="bi-interaction",
                            pool="attn", dropout=0.25, act="leaky_relu",
                            )
        model = WIKGMILModel(config)
    elif model_str == "wikg_mil_roupkt":
        config = WIKGConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes, topk=4,
                            agg_type="bi-interaction",
                            pool="attn", dropout=0.25, act="leaky_relu",
                            enable_roupkt=True,
                            expert_network="MLP",
                            )
        model = WIKGMILModel(config)
    elif model_str == "wikg_mil_rankMidFeat_alphax0.25_lora":
        config = WIKGConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes, topk=4,
                            agg_type="bi-interaction",
                            pool="attn", dropout=0.25, act="leaky_relu",
                            classifier_rank=2, classifier_alpha=0.5,
                            # attn_rank=64, attn_alpha=16,
                            patch_rank=64, patch_alpha=16,
                            # agg_rank=64, agg_alpha=16,
                            gate_rank=32, gate_alpha=8,add_route=True
                            )
        model = WIKGMILModel(config)
    elif model_str == "wikg_mil_rank0.25Feat_alphax0.25_lora":
        config = WIKGConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes, topk=4,
                            agg_type="bi-interaction",
                            pool="attn", dropout=0.25, act="leaky_relu",
                            classifier_rank=1, classifier_alpha=0.25,
                            patch_rank=32, patch_alpha=8,
                            gate_rank=16, gate_alpha=4,
                            )
        model = WIKGMILModel(config)
    elif model_str == "wikg_mil_rank0.75Feat_alphax0.25_lora":
        config = WIKGConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes, topk=4,
                            agg_type="bi-interaction",
                            pool="attn", dropout=0.25, act="leaky_relu",
                            classifier_rank=3, classifier_alpha=0.75,
                            patch_rank=96, patch_alpha=24,
                            gate_rank=48, gate_alpha=12,
                            )
        model = WIKGMILModel(config)
    elif model_str == "wikg_mil_rank1.0Feat_alphax0.25_lora":
        config = WIKGConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes, topk=4,
                            agg_type="bi-interaction",
                            pool="attn", dropout=0.25, act="leaky_relu",
                            classifier_rank=4, classifier_alpha=1,
                            patch_rank=128, patch_alpha=32,
                            gate_rank=64, gate_alpha=16,  # gate的最小矩阵是
                            )
        model = WIKGMILModel(config)
    elif model_str == "mico":
        model = MiCo(survival=True, dim_in=512, drop=0.05, num_enhancers=3)
        model.reset_cluster_center(
            f"/xxx/projects2/fedmed2026/cluster_files/site_{dataset}_fold_{fold_id}_64clusters.pth")
    elif model_str == "mico_rankMidFeat_alphax0.25_lora":
        model = MiCo(survival=True, dim_in=512, drop=0.05, num_enhancers=3,
                          classifier_rank=2, classifier_alpha=0.5,
                          attention_rank=128, attention_alpha=32,
                          patch_cluster_rank=128, patch_cluster_alpha=32,
                          )
        model.reset_cluster_center(
            f"/xxx/projects2/fedmed2026/cluster_files/site_{dataset}_fold_{fold_id}_64clusters.pth")
    else:
        raise NotImplementedError("unsupported model name: {}".format(model_str))

    if model_str.endswith("_lora"):
        load_checkpoint_lora(args.checkpoint, model, use_pretrained_classifier=args.load_pretrain_classifier, only_finetune_classifier=args.only_finetune_classifier,
                             fix_expert=args.fix_expert,
                             no_expert=args.no_expert,
                             no_expert_initial_expert_num=args.no_expert_initial_expert_num,
                             average_route_scores=args.average_route_scores)
    else:
        load_checkpoint(args.checkpoint, model, use_pretrained_classifier=args.load_pretrain_classifier, only_finetune_classifier=args.only_finetune_classifier)

    optimizer, scheduler, router_optimizer = create_optimizer_scheduler(model, args.optimizer, weight_decay=weight_decay,
                                                      backbone_lr=args.feature_lr,
                                                      classifier_lr=args.lr, lora_lr=args.lora_lr, lora_wd=args.lora_wd)
    print("Model info")
    print(model)
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameter size: {params / 1e6:.2f}M")

    if model_str.endswith("_lora"):
        print("Training model")
        tmp_test_metrics = defaultdict(list)
        for epoch in range(max_epochs):
            lrs = [pg["lr"] for pg in  optimizer.param_groups]

            train_lora(model, train_dataloader, optimizer, device, epoch, router_optimizer=router_optimizer)
            if scheduler is not None:
                scheduler.step(epoch)

            c_index, ci_lower, ci_upper, pvalue_pred, results_dict = testting(model, test_dataloader, device, epoch)
            print(
                f"Epoch{epoch}, Test, c_indx: {c_index:.5f}, p_value: {pvalue_pred:.5f}, 95% CI for C-Index: [{ci_lower:.4f}, {ci_upper:.4f}], lrs:{lrs}")
            for k, v in results_dict.items():
                tmp_test_metrics[k].append(v)
            tmp_test_metrics['epoch'].append(epoch)
    else:
        print("Training model")
        tmp_test_metrics = defaultdict(list)
        for epoch in range(max_epochs):
            lrs = [pg["lr"] for pg in  optimizer.param_groups]

            train(model, train_dataloader, optimizer, device, epoch)
            if scheduler is not None:
                scheduler.step(epoch)

            c_index, ci_lower, ci_upper, pvalue_pred, results_dict = testting(model, test_dataloader, device, epoch)
            print(
                f"Epoch{epoch}, Test, c_indx: {c_index:.5f}, p_value: {pvalue_pred:.5f}, 95% CI for C-Index: [{ci_lower:.4f}, {ci_upper:.4f}], lrs:{lrs}")
            for k, v in results_dict.items():
                tmp_test_metrics[k].append(v)
            tmp_test_metrics['epoch'].append(epoch)



    pd.DataFrame(tmp_test_metrics).to_csv(os.path.join(save_dir, f"test_results_fold{fold_id}.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(save_dir, f"model_fold{fold_id}.pth"))
    check_all_existed = all([os.path.exists(os.path.join(save_dir, f"test_results_fold{c}.csv")) for c in range(args.num_fold)])
    if check_all_existed:
        # 加载了所有的
        all_fold_max_values = defaultdict(list)
        all_fold_latest_values = defaultdict(list)

        for loaded_fold_id in range(args.num_fold):
            df = pd.read_csv(os.path.join(save_dir, f"test_results_fold{loaded_fold_id}.csv"))
            for metric_key in df.columns:
                if 'epoch' in metric_key:
                    continue
                max_v = df[metric_key].max()
                latest_v = df[metric_key].iloc[-1]
                all_fold_max_values[metric_key].append(max_v)
                all_fold_latest_values[metric_key].append(latest_v)

        print(f"==> exp_name:{args.exp_name}")
        print(f"All {args.num_fold} Fold Max Values:")
        all_max_value_record = defaultdict(list)
        for metric_key in all_fold_max_values.keys():
            print(
                f"\t{metric_key:<20} - {np.mean(all_fold_max_values[metric_key])} ± {np.std(all_fold_max_values[metric_key])}")
            all_max_value_record['metric_name'].append(metric_key)
            all_max_value_record['avg_value'].append(np.mean(all_fold_max_values[metric_key]))
            all_max_value_record['std'].append(np.std(all_fold_max_values[metric_key]))
        print(f"All {args.num_fold} Fold Latest Values:")
        all_latest_value_record = defaultdict(list)
        for metric_key in all_fold_latest_values.keys():
            print(
                f"\t{metric_key:<20} - {np.mean(all_fold_latest_values[metric_key])} ± {np.std(all_fold_latest_values[metric_key])}")
            all_latest_value_record['metric_name'].append(metric_key)
            all_latest_value_record['avg_value'].append(np.mean(
                all_fold_latest_values[metric_key]))
            all_latest_value_record['std'].append(np.std(all_fold_latest_values[metric_key]))

        pd.DataFrame(all_max_value_record).to_csv(os.path.join(save_dir, f"test_results_avg_folds_max.csv"), index=False)
        pd.DataFrame(all_latest_value_record).to_csv(os.path.join(save_dir, f"test_results_avg_folds_latest.csv"), index=False)

