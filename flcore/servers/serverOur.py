import copy
import os
import time

import torch

from flcore.clients.clientOur import clientOur
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread

from flcore.utils.average_meter import AverageMeter

PRINT_COLLECT_CLIENT_UPLOADED_MODEL_NAMES = False

class OurServer(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientOur)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.global_hypernet = copy.deepcopy(self.clients[0].hypernet)


    def send_models(self):
        res = super().send_models()
        for client in self.clients:
            client.hypernet.load_state_dict(self.global_hypernet.state_dict())
        for client in self.test_global_model_clients.values():
            client.hypernet.load_state_dict(self.global_hypernet.state_dict())
        return res

    def aggregate_parameters(self):
        res = super().aggregate_parameters()
        global_dict = self.global_hypernet.state_dict()
        client_hypernet_dicts = [client.hypernet.state_dict() for client in self.selected_clients]

        for k in global_dict.keys():
            global_dict[k] = torch.stack([
                client_dict[k].float() for client_dict in client_hypernet_dicts
            ], 0).mean(0)

        self.global_hypernet.load_state_dict(global_dict)
        return res

    def train(self,):
        self.send_models()

        for i in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            loss_task_meter = AverageMeter()
            loss_lora_meter = AverageMeter()
            for client in self.selected_clients: # fixme 这么大的bug!!!!! 估计全部重跑!
                loss_task, loss_lora = client.train(i)
                loss_task_meter.update(loss_task, 1)
                loss_lora_meter.update(loss_lora, 1)
            print("Average task loss: {}".format(loss_task_meter.avg))
            print("Average lora loss: {}".format(loss_lora_meter.avg))
            # 这里可以关联到early stop这类的
            # print(f"\n-------------Round number: {i}-------------")
            # print(f"\n[{self.__class__.__name__}] Val Results: ")
            # print("\nEvaluate local model")
            # self.evaluate(i, split_name="val")

            self.receive_models()
            if self.dlg_eval and i %self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()
            self.send_models()

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
        print("Collect prototypes...")
        self.prototypes = dict()
        for client in self.clients:
            self.prototypes[self.train_client_names[client.id]] = client.collect_prototype()
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
        torch.save({
            "global_model": self.global_model.state_dict(),
            'hypernet': self.global_hypernet.state_dict(),
            "prototypes": self.prototypes,
        }, model_path)

    def save_best_model(self, result_dict, eval_key='c_index'):
        if eval_key not in self.best_values:
            self.best_values[eval_key] = -1 # 仅仅支持max模式!
        if result_dict[eval_key] > self.best_values[eval_key]:
            model_path = "models"  # os.path.join("models", self.dataset)

            # model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
            os.makedirs(f"{model_path}/{self.experiment_sub_dirname}", exist_ok=True)
            model_path = f"{model_path}/{self.experiment_sub_dirname}" + self.algorithm + f"_{self.goal}_server_fold{self.fold_id}_best.pt"
            # torch.save(self.global_model.state_dict(), model_path)
            torch.save({
                "global_model": self.global_model.state_dict(),
                "hypernet": self.global_hypernet.state_dict(),
                'prototypes': None
                # "backup_recent_lora_params": self.prototypes,
            }, model_path)

            self.best_values[eval_key] = result_dict[eval_key]

            print(f"Best result for {eval_key}: {self.best_values[eval_key]}, save model to {model_path}")
