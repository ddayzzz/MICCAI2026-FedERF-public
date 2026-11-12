import copy
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from torch import nn, optim

from flcore.clients.clientOur import clientOur
from flcore.clients.clientOur2 import clientOur2
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread

from flcore.trainmodel.hypernet2 import ViTHyper
from flcore.trainmodel.lora_modules.lora_layers import Linear as LoRALinearLayer
from flcore.trainmodel.mil_models.layers import create_mlp
from flcore.utils.average_meter import AverageMeter
from torch.nn.utils import spectral_norm

PRINT_COLLECT_CLIENT_UPLOADED_MODEL_NAMES = False

class Our2Server(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientOur2)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        assert self.join_ratio == 1.0
        # self.load_model()
        self.hyper_lr = 0.001
        self.hyper_opt_name = "adamw"
        self.hyper_wd = 0.0001
        self.Budget = []
        lora_names = []
        lora_shapes = []
        for name, module in self.global_model.named_modules():
            if isinstance(module, LoRALinearLayer):
                lora_names.append(name)
                lora_shapes.append((tuple(module.lora_A.size()), tuple(module.lora_B.size())))
        hyper = ViTHyper(embedding_dim=384, embedding_hidden_dim=256, hidden_dim=384, layer_shapes=lora_shapes, layer_lora_names_prefixs=lora_names, )
        self.hyper = hyper.to(self.device)
        if self.hyper_opt_name == "adamw":
            self.hyper_optimizer = optim.AdamW(self.hyper.parameters(), lr=self.hyper_lr, weight_decay=self.hyper_wd)

    def train(self,):
        self.collected_prototypes = {ci.id: None for ci in self.clients}
        # self.collected_prototypes = {ci.id: torch.rand(384, device=self.device) for ci in self.clients}  # DEBUG
        for i in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            loss_task_meter = AverageMeter()
            loss_lora_meter = AverageMeter()
            collect_client_ids = []

            hn_acc_grads = [torch.zeros_like(k) for k in self.hyper.parameters()] if (all([v is not None for k, v in self.collected_prototypes.items()])) else None


            num_samples= [c.train_samples for c in self.selected_clients]
            grad_weights = np.asarray(num_samples) / sum(num_samples)

            for ci, client in enumerate(self.selected_clients):

                input_prototype = self.collected_prototypes[client.id]
                if input_prototype is None:
                    gen_weight_dict = None
                else:
                    gen_weight_dict = self.hyper(input_prototype)


                loss_task, loss_lora = client.train(i, gen_lora_params=OrderedDict([(k, v.detach())for k, v in gen_weight_dict.items()]) if gen_weight_dict is not None else None)  # detach grad!
                loss_task_meter.update(loss_task, 1)
                loss_lora_meter.update(loss_lora, 1)
                collect_client_ids.append(client.id)
                self.collected_prototypes[client.id] = client.slide_features  # 更新
                # 计算Delta
                if gen_weight_dict is not None:
                    deltas = []
                    st = client.model.state_dict()
                    for name, module in self.global_model.named_modules():
                        if isinstance(module, LoRALinearLayer):
                            delta1 = gen_weight_dict[f"{name}.lora_A"].detach() - st[f"{name}.lora_A"].data
                            delta2 = gen_weight_dict[f"{name}.lora_B"].detach() - st[f"{name}.lora_B"].data
                            deltas.extend([delta1, delta2])

                    hn_grads = torch.autograd.grad(
                        outputs=list(gen_weight_dict.values()),
                        inputs=list(self.hyper.parameters()),
                        grad_outputs=deltas,
                        # retain_graph=ci != (len(self.selected_clients) - 1)
                    )
                    for i, grad in enumerate(hn_grads):
                        hn_acc_grads[i] += (grad * grad_weights[ci])

            print("Average task loss: {}".format(loss_task_meter.avg))
            print("Average lora loss: {}".format(loss_lora_meter.avg))

            self.receive_models()
            if self.dlg_eval and i %self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()
            self.send_models()
            # 更新hyper
            if hn_acc_grads is not None:
                self.hyper_optimizer.zero_grad()
                for gi, (pname, param) in enumerate(self.hyper.named_parameters()):
                    grad = hn_acc_grads[gi]
                    # if grad is None:
                    #     hn_noupdated.append(pname)
                    # else:
                    #     hn_updated.append(pname)
                    param.grad = grad
                self.hyper_optimizer.step()
                for p in self.hyper.parameters():
                    p.grad = None  # 这个一定要设置, 不然gpu会泄露



            if not self.debug and i % self.eval_gap == 0: # 当前需要递进进度
                print(f"\n-------------Round number: {i}-------------")
                print(f"\n[{self.__class__.__name__}] Testing Results: ")
                print("\nEvaluate global model")
                test_current_metrics, test_best_metrics = self.evaluate(i, split_name="test")
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

    def save_global_model(self):
        model_path = "models"  # os.path.join("models", self.dataset)

        # model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        os.makedirs(f"{model_path}/{self.experiment_sub_dirname}", exist_ok=True)
        model_path = f"{model_path}/{self.experiment_sub_dirname}" + self.algorithm + f"_{self.goal}_server_fold{self.fold_id}.pt"

        st = {
            "global": self.global_model.state_dict(),
            'hyper': self.hyper.state_dict(),
            'client_models': dict(),
            "client_prototypes": dict(),
        }
        for client in self.clients:
            st["client_models"][self.train_client_names[client.id]] = client.model.state_dict()
            st["client_prototypes"][self.train_client_names[client.id]] = self.collected_prototypes[client.id]
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

