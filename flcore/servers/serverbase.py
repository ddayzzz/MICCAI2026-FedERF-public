from collections import defaultdict, OrderedDict

import pandas as pd
import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from flcore.utils.dlg import DLG
from typing import OrderedDict as OrderedDictType, Literal
from typing import Tuple, Optional, Union, List
import torch.nn as nn

from flcore.utils.read_client_data import read_client_data


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break
        self.save_gpu = args.save_gpu
        self.num_workers = args.num_workers
        self.seed = args.seed
        self.checkpoint = args.checkpoint
        self.eval_only = args.eval_only
        self.train_client_names = args.train_client_names  # 参与训练的site name
        self.all_client_names = args.all_client_names  # 全局的site name
        # assert len(args.exclude_client_names) == 1
        self.exclude_client_names = args.exclude_client_names
        self.best_values = dict()

        self.debug = os.environ.get("DEBUG", "0") == "1"

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        # 数据集选项
        self.model_str = args.model_str
        self.fold_id = args.base_fold_id + times
        args.fold_id = self.fold_id  # 这里需要设置


        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client


        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

        if not self.save_gpu:
            # move to gpu
            self.global_model = self.global_model.to(self.device)

        if self.exclude_client_names is not None and len(self.exclude_client_names) > 0:
            eval_site_disp = '_'.join(self.exclude_client_names)
            self.experiment_sub_dirname = f"data_{self.dataset}_model_{self.model_str}/eval_site_{eval_site_disp}/"
        else:
            assert self.num_clients == len(self.all_client_names)
            self.experiment_sub_dirname = f"data_{self.dataset}_model_{self.model_str}/eval_all/"
        self.test_global_model_names = args.test_global_model_names
        self.test_global_model_clients = dict()



    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            # 这里不需要设置具体的missing参数设置
            # train_data = read_client_data(self.dataset, self.non_iid_type, i, is_train=True, batch_size=self.batch_size)
            # test_data = read_client_data(self.dataset, self.non_iid_type, i, is_train=False, batch_size=self.batch_size)
            if not self.eval_only:
                train_data = read_client_data(dataset=self.dataset, split="train", site_name=self.train_client_names[i],
                                              batch_size=self.batch_size,
                                              model_name=self.model_str,
                                              num_workers=self.num_workers,
                                              seed=self.seed,
                                              fold_id=self.fold_id)
                n_train_samples = len(train_data.dataset)
            else:
                print("Eval only skip loading training data")
                n_train_samples = 0
            if len(self.test_global_model_names) > 0:
                n_test_data = 0
                print("Skip load local testing data")
            else:
                test_data = read_client_data(dataset=self.dataset, split="test", site_name=self.train_client_names[i],
                                             batch_size=self.batch_size,
                                             model_name=self.model_str,
                                             num_workers=self.num_workers,
                                             seed=self.seed,
                                             fold_id=self.fold_id)
                n_test_data = len(test_data.dataset)

            client = clientObj(self.args, 
                            id=i, 
                            train_samples=n_train_samples,
                            test_samples=n_test_data,
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)
            print(f"client {i}(name={self.train_client_names[i]}), "
                  f"num_train: {n_train_samples}(fold_id={self.fold_id}), "
                  f"num_test: {n_test_data}")
        # 设置指标
        self.train_metric_keys = tuple(client.rs_train_metrics.keys())
        self.val_metric_keys = tuple(client.rs_val_metrics.keys())
        self.test_metric_keys = tuple(client.rs_test_metrics.keys())
        self.rs_test_df = pd.DataFrame([], columns=("client", "round") + self.test_metric_keys)
        for metric in self.test_metric_keys:
            setattr(self, f"rs_test_{metric}", list())

        self.rs_train_df = pd.DataFrame([], columns=("client", "round") + self.train_metric_keys)
        for metric in self.train_metric_keys:
            setattr(self, f"rs_train_{metric}", list())

        self.rs_val_df = pd.DataFrame([], columns=("client", "round") + self.val_metric_keys)
        for metric in self.val_metric_keys:
            setattr(self, f"rs_val_{metric}", list())

        # 增加客户端测试global model
        if len(self.test_global_model_names) > 0:
            for name in self.test_global_model_names:
                print(f"Load global testing name: {name}")
                test_data = read_client_data(dataset=self.dataset, split="test", site_name=name,
                                              batch_size=self.batch_size,
                                              model_name=self.model_str,
                                              num_workers=self.num_workers,
                                              seed=self.seed,
                                             fold_id=self.fold_id)

                client = clientObj(self.args,
                                   id=name,
                                   train_samples=0,
                                   test_samples=len(test_data.dataset),
                                   train_slow=False,
                                   send_slow=False)
                print(f"client {name}, "
                      f"num_test: {len(test_data.dataset)}(fold_id={self.fold_id})")
                self.test_global_model_clients[name] = client


    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
        print(f'Selected clients: {[c.id for c in selected_clients]}')
        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
        for name, client in self.test_global_model_clients.items():
            client.set_parameters(self.global_model)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = self.selected_clients  # random.sample(self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self, rnd):
        assert (len(self.uploaded_models) > 0)
        agg_names = []
        new_st = OrderedDict()
        for k, v in self.global_model.state_dict().items():
            new_st[k] = torch.zeros_like(v)
            agg_names.append(k)
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            for k, v in client_model.state_dict().items():
                new_st[k] += (w * v.data)
        mis_k, unexp_k = self.global_model.load_state_dict(new_st)
        if rnd == 0:
            print(f'Aggregate from clients: {self.uploaded_ids}, with weights: {self.uploaded_weights}, names: {agg_names}')
        else:
            print(f'Aggregate from clients: {self.uploaded_ids}, with weights: {self.uploaded_weights}')

    def save_global_model(self):
        model_path = "models" # os.path.join("models", self.dataset)

        # model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        os.makedirs(f"{model_path}/{self.experiment_sub_dirname}", exist_ok=True)
        model_path = f"{model_path}/{self.experiment_sub_dirname}" + self.algorithm + f"_{self.goal}_server_fold{self.fold_id}.pt"
        torch.save(self.global_model.state_dict(), model_path)

    def load_model(self):
        model_path = "models" # os.path.join("models", self.dataset)
        # os.makedirs(f"{model_path}/{self.experiment_sub_dirname}", exist_ok=True)
        model_path = f"{model_path}/{self.experiment_sub_dirname}" + self.algorithm + f"_{self.goal}_server_fold{self.fold_id}.pt"
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)


    def load_model_from_checkpoint_filepath(self, filepath):
        assert (os.path.exists(filepath))
        self.global_model.load_state_dict(torch.load(filepath, map_location="cpu"))

    def model_exists(self):
        model_path = "models" # os.path.join("models", self.dataset)

        model_path = f"{model_path}/{self.experiment_sub_dirname}" + self.algorithm + f"_{self.goal}_server_fold{self.fold_id}.pt"
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.algorithm
        result_path = f"./fl_results/{self.experiment_sub_dirname}"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        algo = algo + "_" + self.goal + "_fold" + str(self.fold_id)
        file_path = result_path + "{}.h5".format(algo)
        print("File path: " + file_path)
        with h5py.File(file_path, 'w') as hf:
            for metric_key in self.test_metric_keys:
                obj_list = getattr(self, "rs_test_" + metric_key)
                hf.create_dataset(f'rs_test_{metric_key}', data=obj_list)
            # for metric_key in self.val_metric_keys:
            #     obj_list = getattr(self, "rs_val_" + metric_key)
            #     hf.create_dataset(f'rs_val_{metric_key}', data=obj_list)

        self.rs_test_df.to_csv(result_path + "{}_eval_test_results.csv".format(algo))
        # self.rs_val_df.to_csv(result_path + "{}_eval_val_results.csv".format(algo))

    def save_best_model(self, result_dict, eval_key='c_index'):
        if eval_key not in self.best_values:
            self.best_values[eval_key] = -1 # 仅仅支持max模式!
        if result_dict[eval_key] > self.best_values[eval_key]:
            model_path = "models"  # os.path.join("models", self.dataset)

            # model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
            os.makedirs(f"{model_path}/{self.experiment_sub_dirname}", exist_ok=True)
            model_path = f"{model_path}/{self.experiment_sub_dirname}" + self.algorithm + f"_{self.goal}_server_fold{self.fold_id}_best.pt"
            torch.save(self.global_model.state_dict(), model_path)

            self.best_values[eval_key] = result_dict[eval_key]

            print(f"Best result for {eval_key}: {self.best_values[eval_key]}, save model to {model_path}")



    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        if len(self.test_global_model_clients) > 0:
            # ids = [c.id for c in self.test_global_clients]
            print(f"Evaluate global model with global dataset, with ids={self.test_global_model_names}")
            glb_names = []
            num_samples = []
            metrics_avg = defaultdict(list)
            for name, c in self.test_global_model_clients.items():
                test_num, stat_dict = c.test_metrics(data_split="test")
                # tot_correct.append(ct*1.0)
                # tot_auc.append(auc*ns)
                for k, v in stat_dict.items():
                    metrics_avg[k].append(v)
                num_samples.append(test_num)
                glb_names.append(name)
            return glb_names, num_samples, metrics_avg

        num_samples = []
        metrics_avg = defaultdict(list)
        for c in self.clients:
            test_num, stat_dict = c.test_metrics(data_split="test")
            # tot_correct.append(ct*1.0)
            # tot_auc.append(auc*ns)
            for k, v in stat_dict.items():
                metrics_avg[k].append(v)
            num_samples.append(test_num)

        ids = [c.id for c in self.clients]
        client_names = [self.train_client_names[c.id] for c in self.clients]

        return client_names, num_samples, metrics_avg

    def val_metrics(self):

        num_samples = []
        metrics_avg = defaultdict(list)
        for c in self.clients:
            test_num, stat_dict = c.test_metrics(data_split="val")
            for k, v in stat_dict.items():
                metrics_avg[k].append(v)
            num_samples.append(test_num)

        ids = [c.id for c in self.clients]
        client_names = [self.train_client_names[c.id] for c in self.clients]

        return client_names, num_samples, metrics_avg

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        sum_metrics = defaultdict(list)
        for c in self.clients:
            cl, ns, c_metric = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)
            for k, v in c_metric.items():
                sum_metrics[k].append(v)

        ids = [c.id for c in self.clients]
        client_names = [self.train_client_names[c.id] for c in self.clients]

        return client_names, num_samples, losses, sum_metrics

    # evaluate selected clients
    def evaluate(self, rnd, split_name: Literal['val', 'test'] = "test"):
        if split_name == "val":
            stats = self.val_metrics()
            metric_keys = self.val_metric_keys
        elif split_name == "test":
            stats = self.test_metrics()
            metric_keys = self.test_metric_keys
        else:
            raise NotImplementedError

        global_avg_test_metrics = {k: np.mean(v) for k, v in stats[2].items()}
        df_mcs = dict()
        df_mcs['round'] = [rnd] * (len(stats[0]) + 1)   # n_clients + 1 (avg)
        for metric_key in metric_keys:
            print("Averaged {} {}: {}".format(split_name, metric_key, global_avg_test_metrics[metric_key]))
            df_mcs[metric_key] = list(stats[2][metric_key]) + [global_avg_test_metrics[metric_key]]
            df_mcs["client"] = list(stats[0]) + ["average"]
            # 写入到
            getattr(self, f"rs_{split_name}_" + metric_key).append(float(global_avg_test_metrics[metric_key]))

        df = pd.DataFrame(df_mcs)
        if split_name == "val":
            self.rs_val_df = pd.concat([self.rs_val_df, df], axis=0, ignore_index=True)
        else:
            self.rs_test_df = pd.concat([self.rs_test_df, df], axis=0, ignore_index=True)

        return global_avg_test_metrics, self.describe_client_df(split_name=split_name, check_round=rnd)

    def describe_client_df(self, split_name, check_round=None):
        best_values = dict()
        if split_name == "train":
            df = self.rs_train_df
            metric_keys = self.train_metric_keys
            print(f"Train results on round {check_round}")
        elif split_name == "val":
            df = self.rs_val_df
            metric_keys = self.val_metric_keys
            print(f"Val results on round {check_round}")
        else:
            df = self.rs_test_df
            metric_keys = self.test_metric_keys
            print(f"Test results on round {check_round}")
        avg_df = df[df['client'] == "average"]
        # current_df = df[df['round'] == check_round]
        for metric_key in metric_keys:
            if 'loss' in metric_key:
                vi = avg_df[metric_key].argmin()
                rnd = avg_df.iloc[vi]["round"]
                v = avg_df.iloc[vi][metric_key]
                dsp = "Min"
            else:
                vi = avg_df[metric_key].argmax()
                rnd = avg_df.iloc[vi]["round"]
                v = avg_df.iloc[vi][metric_key]
                dsp = "Max"
            best_values[metric_key] = v
            print(f"\t{dsp} {metric_key:<10}: {v} at round {rnd}")
        return best_values



    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        raise NotImplementedError
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            # for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
            for gp, pp in zip(self.get_trainable_parameters(self.global_model), self.get_trainable_parameters(client_model)):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        raise NotImplementedError

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        raise NotImplementedError
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.get_trainable_parameters(client.model), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()


    # evaluating on new clients
    def test_metrics_new_clients(self):
        raise NotImplementedError
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
