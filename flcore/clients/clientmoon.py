import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client


class clientMOON(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.tau = args.tau
        self.mu = args.mu

        self.global_model = None
        self.old_model = copy.deepcopy(self.model)
        if not self.save_gpu:
            self.old_model = self.old_model.to(self.device)
        for p in self.old_model.parameters():
            p.requires_grad = False

    def train(self, rnd, **kwargs):
        self.reset_optimizer(rnd)
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()
        if self.save_gpu:
            self.model.to(self.device)
            self.old_model.to(self.device)

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            loss_sum = 0.0
            for i, batch_dict in enumerate(trainloader):
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
                # rep = self.forward_base(batch_dict)
                # logits = self.forward_head(base_features=rep)
                # loss = self.loss(logits, batch_dict['label'])
                # # rep_old = self.old_model.base(x).detach()
                # rep_old = self.forward_base(batch_dict, model=self.old_model)
                # rep_old = rep_old.detach()
                # # rep_global = self.global_model.base(x).detach()
                # rep_global = self.forward_base(batch_dict, model=self.global_model) # self.global_model.base(x).detach()
                # rep_global = rep_global.detach()
                # loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) / (torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) + torch.exp(F.cosine_similarity(rep, rep_old) / self.tau)))
                # loss += self.mu * torch.mean(loss_con)
                feature_dict = self.forward_base(batch_dict)
                rep = feature_dict['feature_output']
                # print(rep.shape, flush=True)
                output_dict = self.forward_head(batch_dict, feature_dict=feature_dict)

                rep_old_dict = self.forward_base(batch_dict, model=self.old_model)
                rep_old = rep_old_dict['feature_output'].detach()

                rep_glb_dict = self.forward_base(batch_dict, model=self.global_model)
                rep_glb = rep_glb_dict['feature_output'].detach()

                loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, rep_glb) / self.tau) / (
                            torch.exp(F.cosine_similarity(rep, rep_glb) / self.tau) + torch.exp(
                        F.cosine_similarity(rep, rep_old) / self.tau)))
                loss = output_dict['loss']
                loss += self.mu * torch.mean(loss_con)


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()

        # self.model.cpu()
        self.model.zero_grad()

        if self.save_gpu:
            self.model.cpu()
        self.old_model = copy.deepcopy(self.model)
        for p in self.old_model.parameters():
            p.requires_grad = False

        # if self.learning_rate_decay:
        #     self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        return loss_sum / len(trainloader)


    def set_parameters(self, model):
        for (new_name, new_param), (old_name, old_param) in zip(model.state_dict().items(),
                                                                self.model.state_dict().items()):
            # old_param.data = new_param.data.clone()
            # print("src:",ks[i], new_param.shape, ", desc: ", ss[i],old_param.shape)
            old_param.data.copy_(new_param.data)
            # i += 1

        self.global_model = model

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for i, batch_dict in enumerate(trainloader):
                batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
                inputs, rep = self.forward_base(batch_dict)
                outputs, loss = self.forward_head(inputs, rep)
                # loss = self.loss(output, y)

                # rep_old = self.old_model.base(x).detach()
                # rep_global = self.global_model.base(x).detach()
                _, rep_old = self.forward_base(batch_dict, model=self.old_model)
                _, rep_global = self.forward_base(batch_dict, model=self.global_model)
                rep_old = rep_old.detach()
                rep_global = rep_global.detach()
                loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) / (torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) + torch.exp(F.cosine_similarity(rep, rep_old) / self.tau)))
                loss += self.mu * torch.mean(loss_con)
                train_num += batch_dict['label'].shape[0]
                losses += loss.item() * batch_dict['label'].shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num