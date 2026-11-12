import time
from flcore.clients.clientmoon import clientMOON
from flcore.servers.serverbase import Server
from flcore.utils.average_meter import AverageMeter


class MOON(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientMOON)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        self.send_models()
        if self.save_gpu:
            self.global_model.to(self.device)
        for i in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            loss_task_meter = AverageMeter()
            for client in self.selected_clients:
                loss = client.train(i)
                loss_task_meter.update(loss, 1)
            print("Average task loss: {}".format(loss_task_meter.avg))
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
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
            print('-'*50, self.Budget[-1])

            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break

        self.describe_client_df(split_name="test", check_round=self.global_rounds)

        print("\nAveraged time per iteration.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientMOON)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate(self.global_rounds + 1)
