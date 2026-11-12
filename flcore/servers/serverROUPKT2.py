import os
import time

from data.tcga_dataset import TCGADataset
from flcore.clients.clientROUPKT import clientROUPKT
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread

from flcore.utils.average_meter import AverageMeter
import torch

from flcore.utils.read_yaml import read_yaml


class ROUPKT2(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientROUPKT)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.checkpoint = self.checkpoint.format(fold_id=self.fold_id)
        print(f"Load from checkpoint: {self.checkpoint}")
        self.load_model_from_checkpoint_filepath(self.checkpoint)
        self.output_expert_feature_fp = f"roupkt_expert_outputs/{self.model_str}_fold{self.fold_id}"


    def train(self,):
        self.send_models()  # 初始化一下模型
        print("Fine-tuning model, 30 epochs...")
        for i in range(30):
            # 以下是本地finetune
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



        print(f"Average time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        self.describe_client_df(split_name="test", check_round=self.global_rounds)
        print("Inference on experts!")
        # tmp_st = torch.load("/xxx/projects2/fedmed2026/models/data_tcga_model_ab_mil_tiny/eval_site_LIHC/ROUPKT_glb30_loc1_lr0.0002_wd0.00001_optadamw_nc4jr1.0_bs1_noLRdecay_server_fold0.pt", map_location='cpu')
        # for client in self.clients:
        #     client.model.load_state_dict(tmp_st["client_models"][self.train_client_names[client.id]])
        self.inference_target_transfer_data()



    def save_global_model(self):
        model_path = "models"  # os.path.join("models", self.dataset)

        # model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        os.makedirs(f"{model_path}/{self.experiment_sub_dirname}", exist_ok=True)
        model_path = f"{model_path}/{self.experiment_sub_dirname}" + self.algorithm + f"_{self.goal}_server_fold{self.fold_id}.pt"

        st = {
            "global": self.global_model.state_dict(),
            'client_models': dict(),
        }
        for client in self.clients:
            st["client_models"][self.train_client_names[client.id]] = client.model.state_dict()
        torch.save(st, model_path)


    def inference_target_transfer_data(self):
        print(f"Inferent expert on target: {self.exclude_client_names[0]}, save_to: {self.checkpoint}")
        assert len(self.exclude_client_names) == 1
        cfg = read_yaml(f'config/data/{self.exclude_client_names[0]}_survival.yaml')
        for split in ("train", "test"):
            Mydata = TCGADataset(dataset_cfg=cfg.Data, split=split, fold_id=self.fold_id)
            dataloader = torch.utils.data.DataLoader(Mydata, num_workers=4, shuffle=False, batch_size=1)

            for expert_client in self.clients:
                name = self.train_client_names[expert_client.id]
                print("Inference name: {}, split: {}".format(name, split))

                output_dict = dict()
                output_prefix = os.path.join(self.output_expert_feature_fp,
                                             f"transfer_to_{self.exclude_client_names[0]}",
                                             f"expert_{name}_split_{split}.pth")
                os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
                expert_client.model.eval()
                for i, batch_dict in enumerate(dataloader):
                    batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in
                                  batch_dict.items()}
                    slide_id = batch_dict["slide_id"][0]
                    # case_id = batch_dict["case_id"]
                    h, res = expert_client.model.forward_features(h=batch_dict['feature'])
                    output_dict[slide_id] = h.detach().cpu()
                torch.save(output_dict, output_prefix)

