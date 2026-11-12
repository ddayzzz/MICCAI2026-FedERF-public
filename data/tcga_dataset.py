import os.path
import random

import h5py
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import torch.utils.data as data
from typing import List, Tuple, Literal
from flcore.utils.read_yaml import read_yaml


class TCGADataset(data.Dataset):
    def __init__(self,dataset_cfg, fold_id: int,
                 split: Literal["train", "test"],
                 load_experts_names: List[str] = None,
                 search_experts_dir: str = None,
                 n_shot=None,seed=42):
        """

        :param dataset_cfg:
        :param fold_id: fold_id, 从0开始
        :param split: fold_id指定加载那个部分
        """
        self.search_experts_dir = search_experts_dir
        self.load_experts_names = load_experts_names
        self.dataset_cfg = dataset_cfg
        data_csv = dataset_cfg.csv_filepath
        self.fold_id = fold_id
        self.feature_dir = dataset_cfg.feature_dir
        slide_data = pd.read_csv(data_csv)

        test_slide_data = slide_data[slide_data['fold'] == self.fold_id]
        train_slide_data = slide_data[slide_data['fold'] != self.fold_id]
        if split == "train":
            self.slide_data = train_slide_data
        elif split == "test":
            self.slide_data = test_slide_data
        else:
            raise NotImplementedError("Undefined split: {}".format(split))
        self.data = self.slide_data['slide_id'].dropna().reset_index(drop=True)
        self.survival_months = self.slide_data['survival_months'].dropna().reset_index(drop=True)
        self.censorship = self.slide_data['censorship'].dropna().reset_index(drop=True)
        self.case_id = self.slide_data['case_id'].dropna().reset_index(drop=True)
        self.label = self.slide_data['disc_label'].dropna().reset_index(drop=True)

        # ---->Concat related information together
        splits = [self.data, self.survival_months, self.censorship, self.case_id, self.label]
        self.split_data = pd.concat(splits, ignore_index=True, axis=1)
        self.split_data.columns = ['slide_id', 'survival_months', 'censorship', 'case_id', 'disc_label']

        # ---->get patient data
        self.patient_df = self.split_data.drop_duplicates(['case_id']).copy()
        self.patient_df.set_index(keys='case_id', drop=True, inplace=True)
        self.split_data.set_index(keys='case_id', drop=True, inplace=True)

        # ---->Establish a connection between patient_df and data
        self.patient_dict = {}
        dropped_indices = []
        for patient in self.patient_df.index:
            slide_ids = self.split_data.loc[patient, 'slide_id']  # get the case_id
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            #检查是否存在!
            for slide_id in slide_ids:
                if not os.path.exists(os.path.join(self.feature_dir, f"{slide_id}.h5")):
                    dropped_indices.append(patient)
                    continue
            self.patient_dict.update({patient: slide_ids})  # which WSIs are included in each patient

        self.patient_df = self.patient_df.drop(dropped_indices)
        self.patient_df.reset_index(inplace=True)
        # ---->！！！！！for the random mask strategy, get the patient-level label

        if n_shot:
            print(f'Split with {n_shot} shot, with seed: {seed}')
            support_inds = stratified_few_shot_sample(self.patient_df['disc_label'], self.patient_df['censorship'], k_shots=n_shot,
                                                      seed=seed)
            dropped_indices = []
            for patient in self.patient_df.index:
                if patient not in support_inds:
                    dropped_indices.append(patient)

            self.patient_df = self.patient_df.drop(dropped_indices)
            self.patient_df.reset_index(inplace=True)
            self.survival_months = self.patient_df['survival_months']
            self.censorship = self.patient_df['censorship']
            self.case_id = self.patient_df['case_id']
            self.label = self.patient_df['disc_label']
        else:
            self.survival_months = self.patient_df['survival_months']
            self.censorship = self.patient_df['censorship']
            self.case_id = self.patient_df['case_id']
            self.label = self.patient_df['disc_label']


        # 加载专家
        if search_experts_dir is not None:
            print("Will load features from experts dir: {} from cancers: {}".format(search_experts_dir, load_experts_names))
            self.expert_features = {name: torch.load(f"{search_experts_dir}/expert_{name}_split_{split}.pth") for name in load_experts_names}


    def __len__(self):
        return len(self.patient_df)

    def __getitem__(self, idx):
        case_id = self.case_id[idx]
        event_time = self.survival_months[idx]
        censorship = self.censorship[idx]
        label = self.label[idx]
        slide_ids = self.patient_dict[case_id].tolist()

        features = []
        # for slide_id in slide_ids:
        #     full_path = Path(self.feature_dir) / f'{slide_id}.pt'
        #     try:
        #         features.append(torch.load(full_path))
        #     except Exception as e:
        #         print(e, slide_id)
        for slide_id in slide_ids:
            with h5py.File(Path(self.feature_dir) / f'{slide_id}.h5', 'r') as f:
                # 列出文件中所有的 Group 或 Dataset 名称
                # print("Keys in HDF5 file:", list(f.keys()))
                features.append(f['features'][:])

        # return features[0], label, event_time, censorship
        results = {
            "feature": features[0], "label": label, "event_time": event_time, "censorship": censorship, "slide_id": slide_ids[0]
        }
        if self.search_experts_dir is not None:
            concated_exp_features = []
            for name in self.load_experts_names:
                concated_exp_features.append(self.expert_features[name][slide_ids[0]].squeeze(0))
            concated_exp_features = torch.stack(concated_exp_features, dim=0)
            results["expert_features"] = concated_exp_features
        return results


def stratified_few_shot_sample(Y_train, E_train, k_shots=2, seed=42):
    """
    严格按照 Label 和 删失状态 (Censorship) 进行联合分层抽样。

    参数:
    - X_train: 特征矩阵 (n_samples, n_features)
    - Y_train: 离散的标签 (Label，如亚型、分期)
    - E_train: 删失状态 (Censor，通常 1=事件发生, 0=删失)
    - k_shots: 每个 (Label, Censor) 组合层抽取的样本量
    - seed: 随机种子，确保实验可重复
    """
    rng = np.random.default_rng(seed)

    # 构建索引映射，方便按层提取
    df = pd.DataFrame({
        'idx': np.arange(len(Y_train)),
        'label': Y_train,
        'censor': E_train
    })

    support_idx = []

    # 按 Label 和 Censor 联合分组
    for (lbl, cen), group in df.groupby(['label', 'censor']):
        available_indices = group['idx'].values

        # 若某层总样本量小于 k_shots，则全数抽取以防报错
        n_sample = min(k_shots, len(available_indices))
        if n_sample > 0:
            sampled = rng.choice(available_indices, size=n_sample, replace=False)
            support_idx.extend(sampled)

    # 打乱输出结果的顺序
    rng.shuffle(support_idx)

    return support_idx

if __name__ == '__main__':
    cfg = read_yaml('../config/data/LIHC_survival.yaml')
    Mydata = TCGADataset(dataset_cfg=cfg.Data, fold_id=0, split="train",
                         # search_experts_dir="/xxx/projects2/fedmed2026/roupkt_expert_outputs/ab_mil_tiny_fold0/transfer_to_LIHC",
                         # load_experts_names=['SARC', 'STAD', 'SKCM', 'UCEC'],
                         n_shot=32,
                         seed=42,
                         )
    dataloader = data.DataLoader(Mydata)
    for i, data in enumerate(dataloader):
        pass
