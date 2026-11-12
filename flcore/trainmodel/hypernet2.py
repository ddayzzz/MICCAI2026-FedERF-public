from typing import Tuple, List
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

from flcore.trainmodel.mil_models.layers import create_mlp


class ViTHyper(nn.Module):

    def __init__(self, embedding_dim, embedding_hidden_dim, att_hidden_dim, hidden_dim,
                 layer_shapes: List[Tuple[Tuple[int, int],Tuple[int, int]]],
                 layer_lora_names_prefixs: List[str],
                 spec_norm=False, n_hidden=1):
        super(ViTHyper, self).__init__()
        self.layer_shapes = layer_shapes
        self.layer_lora_names_prefixs = layer_lora_names_prefixs
        assert len(layer_lora_names_prefixs) == len(layer_shapes)
        # embedding layer
        self.embeddings = create_mlp(
            in_dim=embedding_dim,
            hid_dims=[embedding_hidden_dim],
            dropout=0.25,
            out_dim=embedding_dim,
            end_with_fc=False
        )
        self.proto_attention = nn.Sequential(
            nn.Linear(embedding_dim, att_hidden_dim),
            nn.Tanh(),
            nn.Linear(att_hidden_dim, 1)
        )

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.to_lora_list = nn.ModuleList([])
        for d in range(len(layer_shapes)):
            lora_dim = layer_shapes[d][0][0] * layer_shapes[d][0][1] + layer_shapes[d][1][0] * layer_shapes[d][1][1]
            to_lora = nn.Linear(hidden_dim, lora_dim)
            self.to_lora_list.append(to_lora)


    def forward(self, x: torch.Tensor):
        """

        :param prototypes: [batch x n_patches x n_dim]
        :return:
        """
        emd = self.embeddings(x)  # [batch x n_patches x n_dim]
        A = self.proto_attention(emd)  # [batch x n_patches x 1]
        A = torch.transpose(A, -2, -1)  # batch x 1 x n_patches
        A = F.softmax(A, dim=-1)  # softmax over N
        h = torch.bmm(A, emd).squeeze(dim=1)  # batch(=1) x n_dim

        pre_shapes = tuple(h.size())[:1]
        assert pre_shapes[0] == 1, 'bs must = 1'

        features = self.mlp(h)
        weights = OrderedDict()
        for d, lora_prefix in enumerate(self.layer_lora_names_prefixs):
            layer_lora_value_hyper = self.to_lora_list[d]
            layer_lora_value = layer_lora_value_hyper(features)
            lora_a_end = self.layer_shapes[d][0][0] * self.layer_shapes[d][0][1]
            lora_a = layer_lora_value[..., :lora_a_end].view(*(pre_shapes + tuple(self.layer_shapes[d][0])))
            lora_b = layer_lora_value[..., lora_a_end:].view(*(pre_shapes + tuple(self.layer_shapes[d][1])))
            weights[f"{self.layer_lora_names_prefixs[d]}.lora_A"] = lora_a[0]
            weights[f"{self.layer_lora_names_prefixs[d]}.lora_B"] = lora_b[0]
        return weights


if __name__ == '__main__':
    hyper = ViTHyper(embedding_dim=512, embedding_hidden_dim=512, hidden_dim=512,layer_shapes=[
        ((512, 8), (8,512)),
    ], layer_lora_names_prefixs=["attnetion.0"], att_hidden_dim=512)

    x = torch.randn((1,44,512))
    w = hyper.forward(x)
    f = 4