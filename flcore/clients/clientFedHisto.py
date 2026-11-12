import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
import os

class clientFedHisto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self, rnd, **kwargs):
        self.reset_optimizer(rnd)
        train_dataloader = self.load_train_data()
        if self.save_gpu:
            self.model.to(self.device)
        # print(os.environ["CUDA_VISIBLE_DEVICES"])
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            loss_sum = 0
            for i, batch_dict in enumerate(train_dataloader):
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
                output_dict = self.forward_model(batch_dict)
                loss = output_dict['loss']
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                self.optimizer.step()
                loss_sum += loss.item()
        if self.save_gpu:
            self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        return loss_sum / len(train_dataloader)
