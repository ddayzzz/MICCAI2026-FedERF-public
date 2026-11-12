# Repository for Cross-cancer Expert-routing Knowledge Transfer in Federated Prognosis Prediction (MICCAI 2026).

##  1. Introduction
Whole-Slide Image (WSI)-based prognosis prediction often struggles to scale to rare cancers due to limited patient cohorts, creating a critical generalization bottleneck. Furthermore, directly sharing knowledge across different cancers via multi-task learning frequently causes negative transfer due to significant prognostic heterogeneity. To address these challenges, we propose a novel Federated Expert-Routing Framework (FedERF) tailored for cross-cancer knowledge transfer. During pre-training, we utilize Federated LoRA across diverse source cancer clients to construct a shared cross-cancer expert pool. This extracts generalizable morphological priors from diverse multi-cancer cohorts, providing robust initialization for data-scarce clients. In the target-adaptive fine-tuning phase, we introduce a dynamic routing mechanism. Instead of applying a rigid global model, this mechanism adaptively selects and aggregates the most relevant experts from the pool based on the target domain's specific semantics. This routing strategy precisely injects complementary knowledge while isolating conflicting features, effectively mitigating negative transfer. Extensive experiments demonstrate that our FedERF successfully overcomes the data scarcity dilemma and achieves superior prognostic performance on target cancers.

## 2. Requirements

The code is implemented in **Python 3.9** and **PyTorch 2.5.1**. To install the required dependencies:

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Data Preparation
FedERF is evaluated on five TCGA cohorts: LIHC, SARC, STAD, SKCM, and UCEC. Feature extraction follows CONCH: 448×448 patches at 20× magnification.

### WSI & Label Processing
To begin, you need to process your Whole Slide Images (WSIs) into feature vectors.

1. For embedding WSIs into patch features, please follow the pipeline outlined in the [CLAM repository](https://github.com/mahmoodlab/CLAM).
2. For processing labels related to survival analysis (follow-up time and censorship), please refer to the methodologies described in [PatchGCN](https://github.com/mahmoodlab/Patch-GCN).
The output of this stage should be a directory of .h5 files, where each file contains the feature vectors for a single WSI.

### Data Split

Change these exact paths inside your `config/data/**.yaml` files:
```yaml
Data:
    dataset_name: TCGADataset
    csv_filepath: ./LIHC_survival.csv
    feature_dir: /path/to/all_suvival_data_4fold/TCGA-LIHC/feats-m20-s448-conch_v1
    train_dataloader:
        batch_size: 1
        num_workers: 8
    test_dataloader:
        batch_size: 1
        num_workers: 8

```

## Training
You can reproduce the experiments by running the provided shell scripts.

- ABMIL:
```bash
# Ours
gpu=1 # CUDA ID
site=LIHC # LIHC, SARC, STAD, SKCM, and UCEC
bash run_transfer.sh our0209 "ab_mil_tiny_rankMidFeat_alphax0.25_lora" ${gpu} 0 4 ${site}
bash run_finetune.sh "fl_our0209_pretrain_noLoadClf_loraClf_loraLast_finetune30Epc_softmaxRt_scale_loraLr1e-4" "ab_mil_tiny_rankMidFeat_alphax0.25_lora" ${gpu} "0 1 2 3" ${site}
# Baselines:
model="ab_mil_tiny"
algo="fedavg"  # fedavg,fedprox,fedhisto,moon
bash run_transfer.sh ${algo} ${model} $gpu 0 4 ${site} &
bash run_finetune.sh "${algo}_pretrain_noLoadClf" ${model} ${gpu} "0 1 2 3" ${site}
# ROUPKT:
model="ab_mil_tiny"
algo="roupkt"  # fedavg,fedprox,fedhisto,moon
bash run_transfer.sh ${algo} ${model} $gpu 0 4 ${site} &
bash run_finetune.sh "roupkt_fedavg_pretrain_noLoadClf" ${model} ${gpu} "0 1 2 3" ${site}
```

- TransMIL:
```bash

gpu=1 # CUDA ID
site=LIHC # LIHC, SARC, STAD, SKCM, and UCEC
bash run_transfer.sh our0209 "trans_mil2_rankMidFeat_alphax0.25_lora" ${gpu} 0 4 ${site}
bash run_finetune.sh "fl_our0209_pretrain_noLoadClf_loraClf_loraLast_finetune30Epc_softmaxRt_scale_loraLr5e-3" "trans_mil2_rankMidFeat_alphax0.25_lora" ${gpu} "0 1 2 3" ${site}
# Baselines:
model="trans_mil2"
algo="fedavg"  # fedavg,fedprox,fedhisto,moon
bash run_transfer.sh ${algo} ${model} $gpu 0 4 ${site} &
bash run_finetune.sh "${algo}_pretrain_noLoadClf" ${model} ${gpu} "0 1 2 3" ${site}
# ROUPKT:
model="trans_mil2"
algo="roupkt"  # fedavg,fedprox,fedhisto,moon
bash run_transfer.sh ${algo} ${model} $gpu 0 4 ${site} &
bash run_finetune.sh "roupkt_fedavg_pretrain_noLoadClf" ${model} ${gpu} "0 1 2 3" ${site}
```

- WiKG:
```bash

gpu=1 # CUDA ID
site=LIHC # LIHC, SARC, STAD, SKCM, and UCEC
bash run_transfer.sh our0209 "wikg_mil_rankMidFeat_alphax0.25_lora" ${gpu} 0 4 ${site}
bash run_finetune.sh "fl_our0209_pretrain_noLoadClf_loraClf_loraLast_finetune30Epc_softmaxRt_scale_loraLr5e-3" "wikg_mil_rankMidFeat_alphax0.25_lora" ${gpu} "0 1 2 3" ${site}
# Baselines:
model="wikg_mil"
algo="fedavg"  # fedavg,fedprox,fedhisto,moon
bash run_transfer.sh ${algo} ${model} $gpu 0 4 ${site} &
bash run_finetune.sh "${algo}_pretrain_noLoadClf" ${model} ${gpu} "0 1 2 3" ${site}
# ROUPKT:
model="wikg_mil"
algo="roupkt"  # fedavg,fedprox,fedhisto,moon
bash run_transfer.sh ${algo} ${model} $gpu 0 4 ${site} &
bash run_finetune.sh "roupkt_fedavg_pretrain_noLoadClf" ${model} ${gpu} "0 1 2 3" ${site}
```

## Directory Structure
- main.py: Main entry for transfer learning
- run_transfer.sh: Main entry for the first phase
- run_finetune.py: Main entry for the second phase
- configs/: YAML configuration files for dataset
- data/: Data loading and interfaces
- flcore/: Federated learning framework
- fl_results: Results for Federated Route learning 
- finetune_results/: Resulting for Fine-tune
