import time

from flcore.clients.clientFedHisto import clientFedHisto
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread

from flcore.utils.average_meter import AverageMeter
import torch
from collections import OrderedDict

class FedHisto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFedHisto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def aggregate_parameters(self, rnd):
        assert (len(self.uploaded_models) > 0)
        default_noise = 0.001  # follow HistoFL setup
        global_w = self.global_model.state_dict()

        for net_id, model in enumerate(self.uploaded_models):
            net_para = model.state_dict()
            agg_names = []
            for key in net_para:
                if net_id == 0:
                    if 'bias' not in key:
                        noise = default_noise * torch.empty(net_para[key].size()).normal_(mean=0,
                                                                                          std=net_para[key].reshape(
                                                                                              -1).float().std())
                        global_w[key] = net_para[key] * self.uploaded_weights[net_id] + noise.to(self.device)
                    else:
                        global_w[key] = net_para[key] * self.uploaded_weights[net_id]
                else:
                    if 'bias' not in key:
                        noise = default_noise * torch.empty(net_para[key].size()).normal_(mean=0,
                                                                                          std=net_para[key].reshape(
                                                                                              -1).float().std())
                        global_w[key] += net_para[key] * self.uploaded_weights[net_id] + noise.to(self.device)
                    else:
                        global_w[key] += net_para[key] * self.uploaded_weights[net_id]
                agg_names.append(key)
        self.global_model.load_state_dict(global_w, strict=False)
        if rnd == 0:
            print(
                f'Aggregate from clients: {self.uploaded_ids}, with weights: {self.uploaded_weights}, names: {agg_names}')
        else:
            print(f'Aggregate from clients: {self.uploaded_ids}, with weights: {self.uploaded_weights}')

    def train(self,):
        self.send_models()
        for i in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            loss_task_meter = AverageMeter()
            for client in self.selected_clients:
                loss = client.train(i)
                loss_task_meter.update(loss, 1)
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
