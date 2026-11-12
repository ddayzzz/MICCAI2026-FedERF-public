from collections import OrderedDict
from typing import Literal

import torch
import torch.nn as nn

class SimpleNoClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.input_dim = input_dim
        blocks = OrderedDict()
        blocks['fc1'] = nn.Linear(input_dim, hidden_dims[0])
        input_dim = hidden_dims[0]
        for li, hidden_dim in enumerate(hidden_dims[1:], start=1):
            blocks['fc' + str(li)] = nn.Linear(input_dim, hidden_dim)
            blocks[f'relu{li}'] = nn.ReLU(inplace=True)
            input_dim = hidden_dim
        self.model = nn.Sequential(blocks)

    def forward(self, x):
        return self.model(x)

class SplitModalNet(nn.Module):
    def __init__(self, input_dim, n_class, hidden_dims, available_modals, fusion_operator: Literal['sum', 'concat']='sum'):
        super().__init__()
        self.fusion_operator = fusion_operator
        self.available_modals = available_modals
        self.n_class = n_class
        self.modal_encoder = nn.ModuleDict()
        for m in available_modals:
            self.modal_encoder[m] = SimpleNoClassifier(input_dim, hidden_dims)
        if self.fusion_operator == "sum":
            output_fusion_dim = hidden_dims[-1]
        elif self.fusion_operator == "concat":
            output_fusion_dim = hidden_dims[-1] * len(available_modals)
        self.clf = nn.Linear(output_fusion_dim, n_class)

    def forward(self, x, input_modals=None):
        if input_modals is None:
            input_modals = self.available_modals
        modal_enc = OrderedDict()
        for modal in input_modals:
             enc = self.modal_encoder[modal](x[modal])
             modal_enc[modal] = enc
        if self.fusion_operator == "concat":
            fused = torch.cat([modal_enc[m] for m in input_modals], dim=1)
        elif self.fusion_operator == "sum":
            fused = sum(modal_enc.values())
        return self.clf(fused)


    def get_modal_encoder(self, modal_name, flatten_weights=False):
        if modal_name not in self.available_modals:
            return None
        enc = self.modal_encoder[modal_name]
        if not flatten_weights:
            return enc.named_parameters()
        flatten_w = None
        for n, p in enc.named_parameters():
            if flatten_w is None:
                flatten_w = p.data.view(-1)
            else:
                flatten_w = torch.cat((flatten_w, p.view(-1)), dim=0)
        return flatten_w

    def get_general_model_weights(self, flatten_weights=False):

        if not flatten_weights:
            return self.clf.named_parameters()
        flatten_w = None
        for n, p in self.clf.named_parameters():
            if flatten_w is None:
                flatten_w = p.data.view(-1)
            else:
                flatten_w = torch.cat((flatten_w, p.view(-1)), dim=0)
        return flatten_w