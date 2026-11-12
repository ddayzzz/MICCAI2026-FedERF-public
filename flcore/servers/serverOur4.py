import time
from collections import defaultdict

from flcore.clients.clientOur4 import clientOur4
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import torch
import os

from flcore.utils.average_meter import AverageMeter


class Our4Server(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientOur4)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.collected_anchors = None


    def train(self,):
        self.send_models()
        for i in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            loss_task_meter = AverageMeter()

            for client in self.selected_clients:
                loss_task = client.train(i)

                loss_task_meter.update(loss_task, 1)


            # self.collected_anchors = proto_aggregation(agg_anchors)
            print("Average task loss: {}".format(loss_task_meter.avg))


            self.receive_models()
            if self.dlg_eval and i %self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters(i)
            self.send_models()

            if i == 0 or (i + 1) % self.eval_gap == 0: # 当前需要递进进度
                print(f"\n-------------Round number: {i}-------------")
                print(f"\n[{self.__class__.__name__}] Testing Results: ")
                print("\nEvaluate global model")
                test_current_metrics, test_best_metrics = self.evaluate(i, split_name="test")
                self.save_best_model(test_current_metrics, eval_key='c_index')


            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break
        print("Finetune LoRA")
        loss_aux_meter = AverageMeter()
        for client in self.clients:
            loss_aux = client.finetune_lora(self.global_rounds * self.local_epochs)
            loss_aux_meter.update(loss_aux, 1)
        print("Average lora loss: {}".format(loss_aux_meter.avg))
        # self.describe_client_df(split_name="val", check_round=self.global_rounds)
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
            'client_models': dict(),
            # "client_prototypes": dict(),
        }
        for client in self.clients:
            st["client_models"][self.train_client_names[client.id]] = client.model.state_dict()
            # st["client_prototypes"][self.train_client_names[client.id]] = self.collected_prototypes[client.id]
        torch.save(st, model_path)