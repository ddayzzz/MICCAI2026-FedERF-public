from typing import Literal

import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from data.tcga_dataset import TCGADataset
from flcore.utils.read_yaml import read_yaml


def read_client_data(dataset, split, site_name, batch_size=None, model_name=None, seed=42, n_fold=4, fold_id=None, num_workers=4):
    if dataset == "tcga":
        cfg = read_yaml(f'config/data/{site_name}_survival.yaml')
        if split == "train":
            Mydata = TCGADataset(dataset_cfg=cfg.Data, fold_id=fold_id, split=split)
            dataloader = DataLoader(Mydata, num_workers=num_workers, shuffle=True, batch_size=1)
        elif split == "val" or split == "test":
            Mydata = TCGADataset(dataset_cfg=cfg.Data, fold_id=fold_id, split=split)
            dataloader = DataLoader(Mydata, num_workers=num_workers, shuffle=False, batch_size=1)
        else:
            raise NotImplementedError("Unsupported split: {}".format(split))
        return dataloader
    else:
        raise ValueError(f"Dataset {dataset} not supported")

