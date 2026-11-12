import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from flcore.clients.clientbase import Client


class clientProx(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu
        self.global_params = [p.clone().detach().cpu() for p in self.model.parameters()]

    def train(self, rnd, **kwargs):
        self.reset_optimizer(rnd)
        trainloader = self.load_train_data()
        start_time = time.time()

        if self.save_gpu:
            self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            loss_sum = 0.0
            for i, batch_dict in enumerate(trainloader):
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
                output_dict = self.forward_model(batch_dict)
                loss = output_dict['loss']
                proximal_term = 0.0
                # 这里修改为求导之前的版本, 只需要使用parameters
                for (n, p), gp in zip(self.model.named_parameters(), self.global_params):
                    proximal_term += (p - gp.data.to(self.device)).norm(2)
                self.optimizer.zero_grad()

                total_loss = loss + proximal_term * self.mu * 0.5
                total_loss.backward()
                if self.grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                # self.optimizer.step(self.global_params, self.device)
                self.optimizer.step()
                loss_sum += total_loss.item()

        if self.save_gpu:
            self.model.cpu()

        # if self.learning_rate_decay:
        #     self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        return loss_sum / len(trainloader)

    def set_parameters(self, model,):
        super().set_parameters(model)
        self.global_params = [p.clone().detach().cpu() for p in model.parameters()]

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        if self.save_gpu:
            self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for i, batch_dict in enumerate(trainloader):
                batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
                outputs = self.forward_model(batch_dict)
                loss = self.loss(outputs, batch_dict['label'])

                bs = batch_dict["label"].shape[0]

                gm = torch.cat([p.data.view(-1) for p in self.global_params], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.get_trainable_parameters(self.model,
                                                                                       state_dict=False)], dim=0)
                loss += 0.5 * self.mu * torch.norm(gm.to(self.device) - pm, p=2)

                self.update_metrics(outputs, batch_dict, is_train=True, loss=loss)

                train_num += bs
                losses += loss.item() * bs
        if self.save_gpu:
            self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num, self.describe_metrics(is_train=True)
