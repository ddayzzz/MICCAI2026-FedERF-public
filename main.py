#!/usr/bin/env python
import copy
import argparse
import os
import time
import warnings
from collections import defaultdict

import numpy as np
import logging

import pandas as pd

from flcore.servers.serverHisto import FedHisto
from flcore.servers.serverOur import OurServer
from flcore.servers.serverOur2 import Our2Server
from flcore.servers.serverOur3 import Our3Server
from flcore.servers.serverOur4 import Our4Server
from flcore.servers.serverROUPKT import ROUPKT
from flcore.servers.serverROUPKT2 import ROUPKT2
from flcore.servers.serveravg import FedAvg
from flcore.servers.servermoon import MOON
from flcore.servers.serverprox import FedProx
from flcore.trainmodel.AB_MIL import AB_MIL, AB_MIL_with_LoRA, AB_MIL_with_LoRA2
from flcore.trainmodel.CACMRN.cross_modal_transformer_v2 import CACMRNModel, get_image_text_fedmmultimodal_CACMRN_model
from flcore.trainmodel.WIKG_MIL import WIKG_MIL
from flcore.trainmodel.generator.autoencoder import get_autoencoder
from flcore.trainmodel.generator.mmr_model import get_image_text_fedmmultimodal_MMR_model
from flcore.trainmodel.mico import MiCo
from flcore.trainmodel.mil_models.abmil import ABMIL, ABMILGatedBaseConfig, ABMILModel
from flcore.trainmodel.mil_models.abmil2 import ABMILModel2
from flcore.trainmodel.mil_models.transmil import TransMILConfig, TransMILModel
from flcore.trainmodel.mil_models.wikg_mil import WIKGMILModel, WIKGConfig
from flcore.trainmodel.mm_models import ImageTextClassifier, UniTextGRUEncoder, ImageTextSeparateClassifier, \
    MMActionClassifier, SERClassifier, SERSeparateClassifiers, MMActionSeparationClassifier

from flcore.trainmodel.models import *
from flcore.trainmodel.ogm_basic_model import OGMAudioFrameModel
from flcore.trainmodel.splitmodalnet import SplitModalNet
from flcore.trainmodel.trans_mil.trans_mil import TRANS_MIL
from flcore.trainmodel.trans_mil.trans_mil2 import TransMIL2

from flcore.trainmodel.transformer import *

from flcore.utils.result_utils import average_data
from flcore.utils.mem_utils import MemReporter
from flcore.servers.serverlocal import Local
import random
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

def set_seeds(seed):
    # 设置随机数的种子
    # fix random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # cudnn
        torch.backends.cudnn.benchmark = False



def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model
    args.model_str = model_str
    eval_results = dict()
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th (fold_id={i}) =============")
        seed = args.seed # 固定 + i
        print(f"Set seed {seed}")
        print(f"All clients: {args.all_client_names}, exclude client names: {args.exclude_client_names}")
        args.train_client_names = [c for c in args.all_client_names if c not in args.exclude_client_names]

        args.seed = seed
        set_seeds(seed)
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "ab_mil": # convex
            # args.model = ABMIL(num_classes=args.num_classes)
            config = ABMILGatedBaseConfig(in_dim=512, embed_dim=384, attn_dim=256, num_classes=args.num_classes)
            model = ABMILModel(config)
            args.model = model
        elif model_str == "ab_mil_tiny":
            config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=64, num_classes=args.num_classes)
            model = ABMILModel(config)
            args.model = model
        elif model_str == "ab_mil_anchor":
            config = ABMILGatedBaseConfig(in_dim=512, embed_dim=384, attn_dim=256, num_classes=args.num_classes)
            args.model = ABMILModel2(config)
        elif model_str == "ab_mil_lora":
            config = ABMILGatedBaseConfig(in_dim=512, embed_dim=384, attn_dim=256, num_classes=args.num_classes, lora_rank=16, lora_alpha=8)
            model = ABMILModel(config)
            args.model = model
        elif model_str == "ab_mil_0204_lora":
            config = ABMILGatedBaseConfig(in_dim=512, embed_dim=384, attn_dim=256, num_classes=args.num_classes,
                                          embedding_rank=16, embedding_alpha=32,
                                          attention_rank=16, attention_alpha=32,
                                          classifier_rank=16, classifier_alpha=32,
                                          final_attention_lora_multi=1, )
            model = ABMILModel(config)
            args.model = model
        elif model_str == "ab_mil_0205_rank4_alpha2_lora":
            config = ABMILGatedBaseConfig(in_dim=512, embed_dim=384, attn_dim=256, num_classes=args.num_classes,
                                          embedding_rank=4, embedding_alpha=2,
                                          attention_rank=4, attention_alpha=2,
                                          classifier_rank=4, classifier_alpha=2,
                                          final_attention_lora_multi=1, )
            model = ABMILModel(config)
            args.model = model
        elif model_str == "ab_mil_0205_rankMidMin_scale0.25_lora":
            config = ABMILGatedBaseConfig(in_dim=512, embed_dim=384, attn_dim=256, num_classes=args.num_classes,
                                          embedding_rank=192, embedding_alpha=48,
                                          attention_rank=32, attention_alpha=8,
                                          classifier_rank=2, classifier_alpha=0.5,
                                          final_attention_lora_multi=1, )
            model = ABMILModel(config)
            args.model = model
        elif model_str == "ab_mil_tiny_rankMidFeat_alphax0.25_lora":
            config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=64, num_classes=args.num_classes,
                                          embedding_rank=64, embedding_alpha=16,
                                          attention_rank=16, attention_alpha=4,
                                          classifier_rank=2, classifier_alpha=0.5,
                                          final_attention_lora_multi=1,
                                          )
            model = ABMILModel(config)
            args.model = model
        elif model_str == "ab_mil_tiny_rankMidFeat_alphax1.0_lora":
            config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=64, num_classes=args.num_classes,
                                          embedding_rank=64, embedding_alpha=64,
                                          attention_rank=16, attention_alpha=16,
                                          classifier_rank=2, classifier_alpha=2,
                                          final_attention_lora_multi=1,
                                          )
            model = ABMILModel(config)
            args.model = model
        elif model_str == "ab_mil_tiny_rankMidFeat_alphax0.5_lora":
            config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=64, num_classes=args.num_classes,
                                          embedding_rank=64, embedding_alpha=32,
                                          attention_rank=16, attention_alpha=8,
                                          classifier_rank=2, classifier_alpha=1,
                                          final_attention_lora_multi=1,
                                          )
            model = ABMILModel(config)
            args.model = model
        elif model_str == "ab_mil_tiny_rankMidFeat_alphax2.0_lora":
            config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=64, num_classes=args.num_classes,
                                          embedding_rank=64, embedding_alpha=128,
                                          attention_rank=16, attention_alpha=32,
                                          classifier_rank=2, classifier_alpha=4,
                                          final_attention_lora_multi=1,
                                          )
            model = ABMILModel(config)
            args.model = model
        elif model_str == "ab_mil_tiny_rank0.25Feat_alphax0.25_lora":
            # 意思是rank为最小维度的1/4
            config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=64, num_classes=args.num_classes,
                                          embedding_rank=32, embedding_alpha=8,
                                          attention_rank=8, attention_alpha=2,
                                          classifier_rank=1, classifier_alpha=0.25,
                                          final_attention_lora_multi=1,
                                          )
            model = ABMILModel(config)
            args.model = model
        elif model_str == "ab_mil_tiny_rank0.75Feat_alphax0.25_lora":
            # 意思是rank为最小维度的3/4
            config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=64, num_classes=args.num_classes,
                                          embedding_rank=96, embedding_alpha=24,
                                          attention_rank=24, attention_alpha=6,
                                          classifier_rank=3, classifier_alpha=0.75,
                                          final_attention_lora_multi=1,
                                          )
            model = ABMILModel(config)
            args.model = model
        elif model_str == "ab_mil_tiny_rank1.0Feat_alphax0.25_lora":
            config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=64, num_classes=args.num_classes,
                                          embedding_rank=128, embedding_alpha=32,
                                          attention_rank=32, attention_alpha=8,
                                          classifier_rank=4, classifier_alpha=1,
                                          final_attention_lora_multi=1,
                                          )
            model = ABMILModel(config)
            args.model = model
        elif model_str == "ab_mil_tiny_rankFullFeat_alphax2_fcNone_lora":
            config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes,
                                          embedding_rank=128, embedding_alpha=256,
                                          attention_rank=32, attention_alpha=64,
                                          classifier_rank=None, classifier_alpha=None,
                                          final_attention_lora_multi=1,
                                          )
            model = ABMILModel(config)
            args.model = model
        elif model_str == "ab_mil_0127_lora":
            args.model = AB_MIL_with_LoRA(num_classes=args.num_classes)

        elif model_str == "trans_mil":
            args.model = TRANS_MIL(num_classes=args.num_classes, dropout=0.1, act=nn.ReLU(), in_dim=512)
        elif model_str == "trans_mil2":
            # args.model = TransMIL2(num_classes=4, dropout=0.25, num_heads=4, num_attention_layers=2, embed_dim=256, in_dim=512, num_fc_layers=1)
            config = TransMILConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes,
                                    num_heads=4, num_fc_layers=1, num_attention_layers=2
                                    )
            model = TransMILModel(config)
            args.model = model
        elif model_str == "trans_mil2_rankMidFeat_alphax0.25_lora":
            # args.model = TransMIL2(num_classes=4, dropout=0.25, num_heads=4, num_attention_layers=2, embed_dim=256, in_dim=512, num_fc_layers=1)
            config = TransMILConfig(in_dim=512, embed_dim=128, num_classes=args.num_classes,
                                    num_heads=4, num_fc_layers=1, num_attention_layers=2,
                                    patch_rank=64, patch_alpha=16,
                                    ppeg_proj_rank=0.5, ppeg_proj_alpha=1,
                                    classifier_rank=2, classifier_alpha=0.5,
                                    )
            model = TransMILModel(config)
            args.model = model
        elif model_str == "wikg_mil":
            # args.model = WIKG_MIL(num_classes=args.num_classes, in_dim=512, dropout=0.25, topk=4, dim_hidden=256)
            config = WIKGConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes, topk=4,
                                agg_type="bi-interaction",
                                pool="attn", dropout=0.25, act="leaky_relu",
                                )
            model = WIKGMILModel(config)
            args.model = model
        #=========== wikg
        elif model_str == "wikg_mil_rankMidFeat_alphax0.25_lora":
            config = WIKGConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes, topk=4,
                                agg_type="bi-interaction",
                                pool="attn", dropout=0.25, act="leaky_relu",
                                classifier_rank=2, classifier_alpha=0.5,
                                patch_rank=64, patch_alpha=16,
                                gate_rank=32, gate_alpha=8,
                                )
            model = WIKGMILModel(config)
            args.model = model
        elif model_str == "wikg_mil_rank0.25Feat_alphax0.25_lora":
            config = WIKGConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes, topk=4,
                                agg_type="bi-interaction",
                                pool="attn", dropout=0.25, act="leaky_relu",
                                classifier_rank=1, classifier_alpha=0.25,
                                patch_rank=32, patch_alpha=8,
                                gate_rank=16, gate_alpha=4,
                                )
            model = WIKGMILModel(config)
            args.model = model
        elif model_str == "wikg_mil_rank0.75Feat_alphax0.25_lora":
            config = WIKGConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes, topk=4,
                                agg_type="bi-interaction",
                                pool="attn", dropout=0.25, act="leaky_relu",
                                classifier_rank=3, classifier_alpha=0.75,
                                patch_rank=96, patch_alpha=24,
                                gate_rank=48, gate_alpha=12,
                                )
            model = WIKGMILModel(config)
            args.model = model
        elif model_str == "wikg_mil_rank1.0Feat_alphax0.25_lora":
            config = WIKGConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=args.num_classes, topk=4,
                                agg_type="bi-interaction",
                                pool="attn", dropout=0.25, act="leaky_relu",
                                classifier_rank=4, classifier_alpha=1,
                                # attn_rank=64, attn_alpha=16,
                                patch_rank=128, patch_alpha=32,
                                # agg_rank=64, agg_alpha=16,
                                gate_rank=64, gate_alpha=16,  # gate的最小矩阵是
                                )
            model = WIKGMILModel(config)
            args.model = model
        #=======
        elif model_str == "mico":
            args.model = MiCo(survival=True, dim_in=512, drop=0.05, num_enhancers=3)
        elif model_str == "mico_rankMidFeat_alphax0.25_lora":
            args.model = MiCo(survival=True, dim_in=512, drop=0.05, num_enhancers=3,
                              classifier_rank=2, classifier_alpha=0.5,
                              attention_rank=128, attention_alpha=32,
                              patch_cluster_rank=128, patch_cluster_alpha=32,
                              )
        else:
            raise NotImplementedError("unsupported model name: {}".format(model_str))

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)
        elif args.algorithm == "FedHisto":
            server = FedHisto(args, i)

        elif args.algorithm == "Local":
            server = Local(args, i)
        elif args.algorithm == "FedProx":
            server = FedProx(args, i)
        elif args.algorithm == "ROUPKT":
            server = ROUPKT(args, i)
        elif args.algorithm == "ROUPKT2":
            server = ROUPKT2(args, i)
        elif args.algorithm == "Our":
            # server = Our2Server(args, i)  # 0130
            # server = Our3Server(args, i)  # 0202
            server = Our4Server(args, i)  # 0203
        elif args.algorithm == "Our0127":
            server = OurServer(args, i)
        else:
            raise NotImplementedError("Please refer to PFLlib repo!")

        if model_str.startswith("mico"):
            print("Initializing client-wise model")
            for client in server.clients:
                sn = client.train_client_names[client.id]
                client.model.reset_cluster_center(f"/xxx/projects2/fedmed2026/cluster_files/site_{sn}_fold_{server.fold_id}_64clusters.pth")
        if args.eval_only:
            print("Evaluating global model on test data!")
            print(f"Load checkpoint file from {args.checkpoint}, with prev={i}")
            # st = torch.load(args.checkpoint.format(prev=i))

            # if args.algorithm == "Local":
            #     for client in server.clients:
            #         client.model.load_state_dict(st[client.id].state_dict())
            # else:
            #     server.global_model.load_state_dict(st.state_dict())
            server.load_model_from_checkpoint_filepath(args.checkpoint.format(prev=i))  # server需要实现不同的加载方式!

            if args.algorithm == "Local":
                pass # 不需要和全局模型同步!
            else:
                server.send_models()
            best_value = server.evaluate(args.global_rounds)
            eval_results[i] = best_value
        else:
            server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    if not args.eval_only:
        # Global average
        average_data(server.test_metric_keys, dataset=args.dataset, algorithm=args.algorithm, goal=args.goal,
                     times=args.times, experiment_sub_dirname=server.experiment_sub_dirname)

    print("All done!")

    reporter.report()
    return eval_results


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="vehicle")
    parser.add_argument('-ncl', "--num_classes", type=int, default=4)
    parser.add_argument('-m', "--model", type=str, default="ab_mil", )
    parser.add_argument('-lbs', "--batch_size", type=int, default=1)

    parser.add_argument('-opn', "--optimizer_name", type=str, default='adam', help="local optimizer", choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.0002, help="Local learning rate")
    parser.add_argument('-lr_end', "--local_learning_rate_end", type=float, default=1e-6, help="Local learning rate")
    parser.add_argument('-wd', "--weight_decay", type=float, default=0.0005, help="Local optimizer weight decay")
    parser.add_argument('-ld', "--lr_decay", type=str, default=None, choices=['linear','cosine'])
    parser.add_argument('-ldg', "--lr_decay_gamma", type=float, default=0.99)
    parser.add_argument('-ldw', "--lr_warmup_round", type=int, default=0)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-sd', "--seed", type=int, default=2025)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100,
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    parser.add_argument("-acn", "--all_client_names", type=str, default=[], nargs="+", help="List of client names")
    parser.add_argument("-ecn", "--exclude_client_names", type=str, default=[], nargs="+", help="List of testing client names (exclude)")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-tstn', "--test_global_model_names", default=[], nargs='+', type=str)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)

    # dataset
    parser.add_argument("--base_fold_id", type=int, default=0, help="Start fold id")
    parser.add_argument('-nk', "--num_workers", type=int, default=2, help="")
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL / FedCAC
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-gradc', "--grad_clip_max_norm", type=float, default=None)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5, help="Server only sends M client models to one client at each round")
    # harmofl
    parser.add_argument("--harmofl_alpha", type=float, default=0.05, help="harmofl alpha")
    # Harmony
    parser.add_argument("--harmony_stage", type=int, choices=[1,2], default=1, help="stage number")
    parser.add_argument("--harmony_stage1_checkpoint", type=str, default='')
    parser.add_argument("--harmony_num_groups", type=int, default=3)
    parser.add_argument("--harmony_stage2_encoder_lr", type=float, default=None)
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, 
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL / IOP-FL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    # MOON / FedCAC / FedLC
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)
    # FedOpt - Adam
    parser.add_argument('-beta1', "--beta1", type=float, default=0.9)
    parser.add_argument('-beta2', "--beta2", type=float, default=0.99)
    parser.add_argument('-epsilon', "--epsilon", type=float, default=0.1)
    # FedSAM
    parser.add_argument('-rho', "--rho", type=float, default=0.1)
    # IOP-FL
    parser.add_argument('-iop_fl_stage', "--iop_fl_stage", type=str, default='fl', choices=['fl', 'odd_test'])

    #  其他的参数
    parser.add_argument("--save_gpu", action="store_true", help="keep local model on cpu when not training while gpu for their turn to train")
    # 加载测试数据集
    parser.add_argument("--eval_only", action="store_true",default=False, help="Only evaluate model")
    parser.add_argument("--eval_algo_name", default=None, type=str, help="evaluation algorithm name, default is algo param")
    parser.add_argument("--checkpoint",default=None, help="checkpoint file")
    parser.add_argument("--save_eval_result_dir",default="results/", help="dir to save eval result")
    parser.add_argument("--eval_missing_rates",default=None, help="only eval missing rates", nargs="+")


    args = parser.parse_args()
    # args.missing_rate = float(args.missing_rate)

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    else:
        args.device = f"cuda:{args.device_id}"
        # args.device = f"cuda"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    if args.eval_only:
        eval_algo_name = args.eval_algo_name
        if eval_algo_name is None:
            eval_algo_name = args.algorithm
        if args.non_iid_type == "dir":
            dataset_dir = f"data_{args.dataset}_dirnon_iid_alpha{args.niid_dir_alpha}"
        res_save_dir = os.path.join(args.save_eval_result_dir, dataset_dir)
        os.makedirs(res_save_dir, exist_ok=True)
        if args.eval_missing_rates is not None and len(args.eval_missing_rates) > 0:
            # 多次修改 missing rate
            best_values = {i:defaultdict(list) for i in range(args.prev, args.times)}
            copyed_model_str = args.model
            for missing_rate in args.eval_missing_rates:
                args.missing_rate = missing_rate
                best_vs = run(args)
                for run_time, best_v in best_vs.items():
                    for k, v in best_v.items():
                        best_values[run_time][k].append(v.item())
                del args.model
                args.model = copyed_model_str
            for rt, best_value in best_values.items():
                dfs = defaultdict(list)

                for k, vs in best_value.items():
                    print(f"Metric: {k}: ")
                    print("missing_rate: ", args.eval_missing_rates, f", values: {vs}")
                    for ms, v in zip(args.eval_missing_rates, vs):
                        dfs['missing_rate'].append(ms)
                        dfs['metric'].append(k)
                        dfs['value'].append(v)
                df = pd.DataFrame(dfs)
                df.to_csv(os.path.join(res_save_dir, f"{eval_algo_name}_{rt}.csv"),
                          index=False)  # 因为多训练了一轮, 因此这里的结果实际上多训练一个round的结果


        else:
            run(args)
    else:
        run(args)

    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
