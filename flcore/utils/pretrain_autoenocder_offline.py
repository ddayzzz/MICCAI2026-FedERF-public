from argparse import ArgumentParser
from typing import List, Tuple

import torch
import torchmetrics
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch import optim
from torchvision import transforms

from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn

# from data.crisismmd_dataset import CrisisMMDDataset
# from flcore.trainmodel.clip_models.clip_missing_aware_prompt_module import MultimodalCLIPModel
# from flcore.trainmodel.clip_models.transforms import CLIP_transform
# from flcore.utils.my_metrics import F1_Score
from torchmetrics import Accuracy, F1Score

from flcore.trainmodel.generator.autoencoder import MultimodalAutoencoder, get_autoencoder
from flcore.trainmodel.mm_models import MMActionClassifier, SERClassifier, ImageTextClassifier
from flcore.utils.read_client_data import read_client_data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def wrap_train_dataset(dataset, site, missing_type, missing_rate,
                 non_iid_type, niid_dir_alpha, niid_pat_class_per_client, fold_id,
                 seed, ):
    train_data = read_client_data(dataset=dataset, idx=site,
                                  is_train=True,
                                  batch_size=batch_size,
                                  return_dataloader=False,
                                  missing_type=missing_type,
                                  missing_rate=missing_rate,
                                  model_name="autoencoder",
                                  num_workers=num_workers,
                                  non_iid_type=non_iid_type,
                                  niid_dir_alpha=niid_dir_alpha,
                                  niid_pat_class_per_client=niid_pat_class_per_client,
                                  seed=seed,
                                  dataset_kws=dict(),
                                  force_missing_config_filename=None,
                                  fold_id=fold_id, )
    # test_data = read_client_data(dataset=dataset, idx=site,
    #                               is_train=False,
    #                               batch_size=batch_size,
    #                               return_dataloader=False,
    #                               missing_type=missing_type,
    #                               missing_rate=missing_rate,
    #                               model_name="autoencoder",
    #                               num_workers=num_workers,
    #                               non_iid_type=non_iid_type,
    #                               niid_dir_alpha=niid_dir_alpha,
    #                               niid_pat_class_per_client=niid_pat_class_per_client,
    #                               seed=seed,
    #                               dataset_kws=dict(),
    #                               force_missing_config_filename=None,
    #                               fold_id=fold_id, )
    return train_data

def wrap_test_dataset(dataset, site, missing_type, missing_rate,
                 non_iid_type, niid_dir_alpha, niid_pat_class_per_client, fold_id,
                 seed, ):
    test_data = read_client_data(dataset=dataset, idx=site,
                                  is_train=False,
                                  batch_size=batch_size,
                                  return_dataloader=False,
                                  missing_type=missing_type,
                                  missing_rate=missing_rate,
                                  model_name="autoencoder",
                                  num_workers=num_workers,
                                  non_iid_type=non_iid_type,
                                  niid_dir_alpha=niid_dir_alpha,
                                  niid_pat_class_per_client=niid_pat_class_per_client,
                                  seed=seed,
                                  dataset_kws=dict(),
                                  force_missing_config_filename=None,
                                  fold_id=fold_id, )
    return test_data

def train(model, train_dataloader, val_dataloader, optimizer, dataset, device, args):
    model.train()
    model.to(device)
    if dataset == "crisis_mmd":
        metric = F1Score(task="multiclass", average="macro", num_classes=8)
        metric_name = "F1_Macro"
    elif dataset == "crema_d" or dataset == "meld" or dataset == "crema_d_v2":
        metric = torchmetrics.Recall(
            task="multiclass",
            average='macro',
            num_classes=args.num_classes
        )
        metric_name = "UAR"
    elif dataset == "ave":
        metric = F1Score(task="multiclass", average="macro", num_classes=28)
        metric_name = "F1_Macro"
    elif dataset == "hateful_memes":
        metric = torchmetrics.AUROC(task="multiclass", num_classes=2)
        metric_name = "auroc"
    metric = metric.to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.max_epochs):
        model.train()
        sum_loss = 0
        for batch_dict in train_dataloader:
            batch_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
            output = model(batch_dict)
            loss = criterion(output, batch_dict['label'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()


        print(f"Epoch[{epoch:<5}], train loss: {sum_loss / len(train_dataloader):.4f}")
        # 测试阶段
        if val_dataloader is not None:
            with torch.no_grad():
                model.eval()
                for batch_dict in val_dataloader:
                    batch_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
                    output = model(batch_dict)
                    metric.update(output, batch_dict['label'])
            metric_v = metric.compute()
            metric.reset()
            print(f"Epoch[{epoch:<5}], test {metric_name}: {metric_v:.4f}")

def train_autoencoder(model, autoencoder, train_dataloader, optimizer, dataset, device, args):
    model.eval()
    model.to(device)
    autoencoder.train()
    autoencoder.to(device)
    if dataset == "crisis_mmd" or dataset == "hateful_memes":
        modal_names = ['image', "text"]
    elif dataset == "crema_d" or dataset == "crema_d_v2" or dataset == "ave":
        modal_names = ['frame', "audio"]

    elif dataset == "meld":
        modal_names = ['text', "audio"]


    for epoch in range(args.max_epochs):
        autoencoder.train()
        sum_loss = 0
        for batch_dict in train_dataloader:
            batch_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
            with torch.no_grad():
                for modal_name in modal_names:
                    batch_dict[f"{modal_name}_feature"] = model.encode_feature(batch_dict, modal_name=modal_name)
            loss_dict = autoencoder(batch_dict)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()


        print(f"Epoch[{epoch:<5}], train loss: {sum_loss / len(train_dataloader):.4f}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--max_train_autoencoder_epochs", type=int, default=200)
    parser.add_argument("--train_data_frac", type=int, default=0.2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--missing_rate", type=float, default=0.0)
    parser.add_argument("--missing_type", type=str, default="train_full_test_both")
    parser.add_argument("--dataset", type=str, default='crema_d')
    parser.add_argument("--site", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--fold_id", type=int, default=None)
    parser.add_argument("--non_iid_type", type=str, default="kfold")
    parser.add_argument("--niid_dir_alpha", type=float, default=None)
    parser.add_argument("--niid_class_per_client", type=int, default=None)
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--base_seed", type=int, default=2025)
    parser.add_argument("--run_id", type=int, default=0)


    args = parser.parse_args()
    seed = args.base_seed + args.run_id
    setup_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    max_epochs = args.max_epochs
    max_train_autoencoder_epochs = args.max_train_autoencoder_epochs
    learning_rate = args.lr
    weight_decay = args.weight_decay
    dataset = args.dataset
    batch_size = args.batch_size
    non_iid_type = args.non_iid_type
    num_workers = 8

    site = args.site
    global_test_site = None


    if site is None:
        site = "all"
    if args.dataset == 'crema_d':
        checkpoint_monitor = "val_uar"
        checkpoint_mode = "max"
        if args.fold_id == 1:
            num_sites = 72
        else:
            num_sites = 73
        base_model_str = "mm_audio_video_action_clf"
        optimizer_name = "sgd"
        global_test_site = "global"
    elif args.dataset == 'meld':
        checkpoint_monitor = "val_uar"
        checkpoint_mode = "max"
        num_sites = 86
        base_model_str = "mm_audio_text_ser_clf"
        optimizer_name = "sgd"
        global_test_site = "global"
    elif args.dataset == 'hateful_memes':
        checkpoint_monitor = "val_auroc"
        checkpoint_mode = "max"
        num_sites = 50
        base_model_str = "mm_img_txt_clf"
        optimizer_name = "sgd"
        global_test_site = None
    elif args.dataset == 'crema_d_v2':
        checkpoint_monitor = "val_uar"
        checkpoint_mode = "max"
        num_sites = 20
        base_model_str = "mm_audio_video_action_clf"
        optimizer_name = "sgd"
        global_test_site = "global"
    elif args.dataset == 'ave':
        checkpoint_monitor = "val_F1_Macro"
        checkpoint_mode = "max"
        num_sites = 20
        base_model_str = "mm_audio_video_action_clf"
        optimizer_name = "sgd"
        global_test_site = "global_test"
    elif args.dataset == 'crisis_mmd':
        checkpoint_monitor = "val_F1_Macro"
        checkpoint_mode = "max"
        num_sites = 20
        base_model_str = "mm_img_txt_clf"
        optimizer_name = "sgd"
        global_test_site = "global_test"



    lr_monitor = LearningRateMonitor(logging_interval='step')


    # logger = TensorBoardLogger(save_dir="pcb_single", name="VisA")
    if args.non_iid_type == "kfold":
        datasuffix = f"noniidtype_{args.non_iid_type}_fold_{args.fold_id}"
    elif args.non_iid_type == "natural":
        datasuffix = f"noniidtype_{args.non_iid_type}_runtime{args.run_id}"
    elif args.non_iid_type == "dir":
        datasuffix = f"noniidtype_{args.non_iid_type}_alpha{args.niid_dir_alpha}_runtime{args.run_id}"
    save_dir = (f"pretrained_autoencoder/single_mixed_{args.dataset}_{datasuffix}_bs{batch_size}_lr{learning_rate}_missingRate_{args.missing_rate}"
                f"")
    os.makedirs(save_dir, exist_ok=True)


    if site == "all":
        trs = []
        for si in range(num_sites):
            train_dataset = wrap_train_dataset(dataset=dataset, site=si, missing_rate=args.missing_rate,
                                               non_iid_type=non_iid_type, niid_dir_alpha=args.niid_dir_alpha,
                                               niid_pat_class_per_client=args.niid_class_per_client,
                                               fold_id=args.fold_id, seed=seed, missing_type=args.missing_type, )
            trs.append(train_dataset)
        collate_fn = train_dataset.get_collate_fn()
        train_dataset = ConcatDataset(trs)

    else:
        train_dataset = wrap_train_dataset(dataset=dataset, site=site,
                                           missing_rate=args.missing_rate,
                                           non_iid_type=non_iid_type,
                                           niid_dir_alpha=args.niid_dir_alpha,
                                           niid_pat_class_per_client=args.niid_class_per_client,
                                           fold_id=args.fold_id, seed=seed,
                                           missing_type=args.missing_type, )
        collate_fn = train_dataset.get_collate_fn()

    if global_test_site is not None:
        test_dataset = wrap_test_dataset(dataset=dataset, site=global_test_site,
                                         missing_rate=args.missing_rate,
                                         non_iid_type=non_iid_type,
                                         niid_dir_alpha=args.niid_dir_alpha,
                                         niid_pat_class_per_client=args.niid_class_per_client,
                                         fold_id=args.fold_id, seed=seed,
                                         missing_type=args.missing_type, )
    else:
        tes = [wrap_test_dataset(dataset=dataset, site=str(si),
                                 missing_rate=args.missing_rate,
                                 non_iid_type=non_iid_type,
                                 niid_dir_alpha=args.niid_dir_alpha,
                                 niid_pat_class_per_client=args.niid_class_per_client,
                                 fold_id=args.fold_id, seed=seed,
                                 missing_type=args.missing_type, ) for si in range(num_sites)]
        test_dataset = ConcatDataset(tes)
    print(f"Use: {args.train_data_frac} of whole training dataset(N={len(train_dataset)})!")
    sub_indices = torch.randperm(len(train_dataset))[:int(len(train_dataset) * args.train_data_frac)]

    train_dataset = Subset(train_dataset, sub_indices)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    print(f"Num of train size: {len(train_dataset)}, Num of test size: {len(test_dataset)}")
    text_dim = {"crisis_mmd": 768, 'hateful_memes': 512, 'food101': 512}
    audio_dim = {'crema_d': 80, 'crema_d_v2': 80, 'ave': 1004}
    if base_model_str == "mm_audio_video_action_clf":
        base_model = MMActionClassifier(
            num_classes=args.num_classes,
            audio_input_dim=audio_dim[dataset],  # mfcc
            video_input_dim=1280,  # mobile
            d_hid=256,
            en_att=False,  # fedmulti KDD, 关闭fedmm的特性
            att_name='',
            clf_hidden_size=128
        )
    elif base_model_str == "mm_audio_text_ser_clf":
        base_model = SERClassifier(
                    num_classes=args.num_classes,
                    audio_input_dim=80, # mfcc
                    text_input_dim=512,  # mobile bert
                    d_hid=256,
                    en_att=False,  # fedmulti KDD, 关闭fedmm的特性
                    att_name='',
                    clf_hidden_size=128
                )
    elif base_model_str == "mm_img_txt_clf":
        base_model = ImageTextClassifier(
                num_classes=args.num_classes,
                img_input_dim=1280,  # mobilenet
                text_input_dim=text_dim[dataset], # mobile_bert
                d_hid=256,
                en_att=False,# fedmulti KDD, 关闭fedmm的特性
                att_name='',
                clf_hidden_size=128
            )
    if optimizer_name == "adam":
        optimizer_fn = lambda xx: torch.optim.Adam(xx.parameters(), weight_decay=weight_decay, lr=learning_rate, betas=(0.5, 0.999), )
    elif optimizer_name == "sgd":
        optimizer_fn = lambda xx: torch.optim.SGD(xx.parameters(), weight_decay=weight_decay, lr=learning_rate, momentum=0.9)

    generator = get_autoencoder(dataset, d_hid=256)

    print("Training encoders!")
    train(base_model, train_dataloader=train_dataloader,
          val_dataloader=test_dataloader,
          # val_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn),
          optimizer=optimizer_fn(base_model), device="cuda", args=args,
          dataset=args.dataset, )
    print("Training autoencoders!")
    train_autoencoder(base_model, autoencoder=generator, train_dataloader=train_dataloader,optimizer=optimizer_fn(generator), device="cuda", args=args,
          dataset=args.dataset, )

    torch.save(base_model, f=f"{save_dir}/base_model.pt")
    torch.save(generator, f=f"{save_dir}/autoencoder.pt")
