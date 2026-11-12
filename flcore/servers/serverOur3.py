import copy
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from torch import nn, optim

from flcore.clients.clientOur import clientOur
from flcore.clients.clientOur2 import clientOur2
from flcore.clients.clientOur3 import clientOur3, get_flat_params_from, set_flat_params_to
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread

from flcore.trainmodel.hypernet2 import ViTHyper
from flcore.trainmodel.lora_modules.lora_layers import Linear as LoRALinearLayer
from flcore.trainmodel.mil_models.layers import create_mlp
from flcore.utils.average_meter import AverageMeter
from torch.nn.utils import spectral_norm

PRINT_COLLECT_CLIENT_UPLOADED_MODEL_NAMES = False

class Our3Server(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientOur3)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        assert self.join_ratio == 1.0
        self.Budget = []
        # self.load_model()
        self.hyper_server_lr = 1.0
        self.hypernet = copy.deepcopy(self.clients[0].hyper)
        for client in self.clients:
            client.hyper.load_state_dict(self.hypernet.state_dict())  # 确保一致
        print("Hyper net:")
        print(self.hypernet)


    def train(self,):
        for rnd in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            loss_task_meter = AverageMeter()
            loss_lora_meter = AverageMeter()
            collect_client_ids = []


            num_samples= [c.train_samples for c in self.selected_clients]
            grad_weights = np.asarray(num_samples) / sum(num_samples)

            hyper_vecs = []
            org_hn_param_vec = copy.deepcopy(get_flat_params_from(self.hypernet))
            for ci, client in enumerate(self.selected_clients):
                loss_task, loss_lora = client.train(rnd)
                loss_task_meter.update(loss_task, 1)
                loss_lora_meter.update(loss_lora, 1)
                collect_client_ids.append(client.id)
                hyper_vecs.append(org_hn_param_vec - client.pseudo_hyper_gradient)


            print("Average task loss: {}".format(loss_task_meter.avg))
            print("Average lora loss: {}".format(loss_lora_meter.avg))

            self.receive_models()
            if self.dlg_eval and rnd %self.dlg_gap == 0:
                self.call_dlg(rnd)

            self.aggregate_parameters()
            # 更新hyper
            # 1. 随机打乱顺序 (PCGrad 对顺序敏感，打乱有助于随机性)

            # 存储修正后的梯度
            projected_grads = []

            for i in range(len(hyper_vecs)):
                g_i = hyper_vecs[i]
                g_i_original = g_i.clone()  # 备份一下，用于投影计算

                # 与其他所有梯度进行两两对比
                for j in range(len(hyper_vecs)):
                    if i == j: continue

                    g_j = hyper_vecs[j]

                    # 计算内积 (Cosine Similarity 的分子)
                    inner_dot = torch.dot(g_i, g_j)

                    # 如果内积 < 0，说明方向冲突 (夹角 > 90度)
                    if inner_dot < 0:
                        # 投影公式: g_i = g_i - ( (g_i * g_j) / (g_j * g_j) ) * g_j
                        # 即: 去掉 g_i 中在 g_j 反方向上的分量
                        denom = torch.dot(g_j, g_j) + 1e-8  # 防止除零

                        # 减去冲突分量
                        g_i = g_i - (inner_dot / denom) * g_j

                projected_grads.append(g_i)
            #
            # # 2. 聚合修正后的梯度
            # # 通常取平均
            final_update = torch.stack(projected_grads).mean(dim=0)

            # old_params = get_flat_params_from(self.hypernet)
            new_params = org_hn_param_vec - self.hyper_server_lr * final_update
            # new_params = torch.zeros_like(hyper_vecs[0])
            # for ww, hp in zip(grad_weights, hyper_vecs):
            #     new_params += (ww * hp)
            set_flat_params_to(self.hypernet, new_params)

            self.send_models()

            if not self.debug and rnd % self.eval_gap == 0: # 当前需要递进进度
                print(f"\n-------------Round number: {rnd}-------------")
                print(f"\n[{self.__class__.__name__}] Testing Results: ")
                print("\nEvaluate global model")
                test_current_metrics, test_best_metrics = self.evaluate(rnd, split_name="test")
                self.save_best_model(test_current_metrics, eval_key='c_index')


            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])



        self.describe_client_df(split_name="test", check_round=self.global_rounds)

        print(f"Average time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()



        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate(self.global_rounds+1)

    def send_models(self):
        res = super().send_models()
        for client in self.clients:
            client.hyper.load_state_dict(self.hypernet.state_dict())
        for name, client in self.test_global_model_clients.items():
            client.hyper.load_state_dict(self.hypernet.state_dict())
        return res

    def save_global_model(self):
        model_path = "models"  # os.path.join("models", self.dataset)

        # model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        os.makedirs(f"{model_path}/{self.experiment_sub_dirname}", exist_ok=True)
        model_path = f"{model_path}/{self.experiment_sub_dirname}" + self.algorithm + f"_{self.goal}_server_fold{self.fold_id}.pt"

        st = {
            "global": self.global_model.state_dict(),
            'hyper': self.hypernet.state_dict(),
            'client_models': dict(),
            "client_prototypes": dict(),
        }
        # for client in self.clients:
        #     st["client_models"][self.train_client_names[client.id]] = client.model.state_dict()
        #     st["client_prototypes"][self.train_client_names[client.id]] = self.collected_prototypes[client.id]
        torch.save(st, model_path)




def get_lora_param_dict(lora_model):
    params = OrderedDict()
    for k, v in lora_model.named_parameters():
        if "lora_A" in k or "lora_B" in k:
            params[k] = v.detach().clone()
    return params

def calculate_euclidean_distance(a, b):
    euclidean_distance = torch.norm(torch.abs(a - b))
    return euclidean_distance


def get_sim_score(cur_lora, other_lora):
    total_sim = 0
    for w1, w2 in zip(cur_lora, other_lora):
        dis = calculate_euclidean_distance(w1, w2.detach().clone())
        sim = 1 / torch.exp(dis)
        total_sim += sim
    total_sim /= len(cur_lora)
    return total_sim

