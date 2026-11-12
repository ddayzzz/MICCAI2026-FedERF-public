import os
import time

import torch

from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread


class Local(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        # self.load_model()
        self.Budget = []


    def train(self):
        self.send_models()
        for i in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()


            for client in self.selected_clients:
                client.train(i)

            print(f"\n-------------Round number: {i}-------------")
            print(f"\n[{self.__class__.__name__}] Val Results: ")
            print("\nEvaluate personalized models")
            self.evaluate(i, split_name='val')

            if not self.debug and i % self.eval_gap == 0: # 当前需要递进进度
                print(f"\n[{self.__class__.__name__}] Testing Results: ")
                print("\nEvaluate personalized model")
                self.evaluate(i, split_name="test")

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break


        # print("\nBest accuracy.")
        # # self.print_(max(self.rs_test_acc), max(
        # #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(self.rs_test_acc))
        # print("\nAverage time cost per round.")
        # print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.describe_client_df(split_name="val", check_round=self.global_rounds)
        self.describe_client_df(split_name="test", check_round=self.global_rounds)
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def save_global_model(self):
        model_path = "models"  # os.path.join("models", self.dataset)

        # model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        os.makedirs(f"{model_path}/{self.experiment_sub_dirname}", exist_ok=True)
        model_path = f"{model_path}/{self.experiment_sub_dirname}" + self.algorithm + f"_{self.goal}_client_models.pt"
        # torch.save(self.global_model, model_path)
        st = dict()
        for client in self.clients:
            st[client.id] = client.model
        torch.save(st, model_path)

    def load_model_from_checkpoint_filepath(self, filepath):
        assert (os.path.exists(filepath))
        sts = torch.load(filepath)
        for client in self.clients:
            client.model.load_state_dict(sts[client.id].state_dict())