import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib
from dataclasses import dataclass

from flcore.trainmodel.mil_models.layers import GlobalAttention, GlobalGatedAttention, create_mlp
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GlobalAttention as GeoGlobalAttention
from flcore.trainmodel.mil_models.mil_template import MIL
from flcore.trainmodel.lora_modules.lora_layers import Linear, LoRALayer
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel
from typing import  List, Literal
from flcore.trainmodel.route_modules import ProtoGuidedLoRA


def get_act(act: str):
    if act == 'leaky_relu':
        return nn.LeakyReLU()
    elif act == 'relu':
        return nn.ReLU()
    elif act == 'tanh':
        return nn.Tanh()
    # Add other activations as needed
    else:
        raise NotImplementedError


# --- 1. Configuration Class ---
@dataclass
class WIKGConfig(PretrainedConfig):
    model_type = 'wikg'

    def __init__(self,
                 in_dim: int = 1024,
                 embed_dim: int = 512,
                 num_classes: int = 2,
                 topk: int = 4,
                 agg_type: str = 'bi-interaction',
                 pool: str = 'attn',
                 dropout: float = 0.25,
                 act: str = 'leaky_relu',
                 lora_rank: int = None,
                 lora_alpha: int = None,
                 add_route: bool = False,
                 classifier_alpha=None, classifier_rank=None,
                 patch_rank=None, patch_alpha=None,
                 gate_rank=None, gate_alpha=None,
                 attn_rank=None, attn_alpha=None,
                 agg_rank=None, agg_alpha=None,
                 # ROUPKT
                 enable_roupkt=False,
                 expert_size=4, expert_network: Literal['Identity', 'MLP'] = "Identity", expert_topk=2,

                 **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.topk = topk
        self.agg_type = agg_type
        self.pool = pool
        self.dropout = dropout
        self.act = act
        #
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.add_route = add_route
        self.classifier_alpha = classifier_alpha
        self.classifier_rank = classifier_rank
        self.patch_rank, self.patch_alpha = patch_rank, patch_alpha
        self.gate_rank, self.gate_alpha = gate_rank, gate_alpha
        self.attn_rank, self.attn_alpha = attn_rank, attn_alpha
        self.agg_rank, self.agg_alpha = agg_rank, agg_alpha
        self.expert_network = expert_network
        self.expert_topk = expert_topk
        self.enable_roupkt = enable_roupkt
        self.expert_size = expert_size

# --- 2. Core WIKG-MIL Model ---
class WIKGMIL(MIL):
    def __init__(self, in_dim: int = 1024, embed_dim: int = 512, num_classes: int = 2, agg_type: str = 'bi-interaction',
                 pool: str = 'attn', dropout: float = 0.25, act: str = 'leaky_relu', topk: int = 6,
                 lora_rank: int = None,
                 lora_alpha: int = None,
                 add_route: bool = False,
                 prefix_name: str = '',
                 classifier_rank=None, classifier_alpha=None,
                 patch_rank=None, patch_alpha=None,
                 gate_rank=None, gate_alpha=None,
                 attn_rank=None, attn_alpha=None,
                 agg_rank=None, agg_alpha=None,
                 # ROUPKT
                 enable_roupkt=False,
                 expert_size=4, expert_network: Literal['Identity', 'MLP'] = "Identity", expert_topk=2,
                 **kwargs):
        """
        Initializes the WIKGMIL model.

        Args:
            in_dim (int): Input dimension of node features.
            embed_dim (int): Embedding dimension for node features.
            num_classes (int): Number of output classes.
            agg_type (str): Type of aggregation to use ('gcn', 'sage', or 'bi-interaction').
            pool (str): Type of pooling to use ('mean', 'max', or 'attn').
            dropout (float): Dropout rate.
            act (str): Activation function to use ('leaky_relu', 'relu', or 'tanh').
            topk (int): Number of top-k nodes to consider for attention.
            **kwargs: Additional keyword arguments.
        """
        self.agg_type = agg_type
        self.topk = topk
        self.pool = pool
        self.dropout = dropout
        self.act = act
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.agg_type = agg_type
        self.pool = pool
        self.dropout = dropout
        self.act = act
        self.n_experts = expert_size
        self.expert_network = expert_network
        self.expert_topk = expert_topk
        self.enable_roupkt = enable_roupkt

        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        for k, v in kwargs.items():
            setattr(self, k, v)
        if classifier_alpha is None:
            classifier_alpha = lora_alpha
            classifier_rank = lora_rank
        if patch_rank is None:
            patch_rank = lora_rank
            patch_alpha = lora_alpha
        if gate_rank is None:
            gate_alpha = lora_alpha
            gate_rank = lora_rank
        if attn_rank is None:
            attn_rank = lora_rank
            attn_alpha = lora_alpha
        if agg_rank is None:
            agg_rank = lora_rank
            agg_alpha = lora_alpha

        dim_hidden = embed_dim
        patch_linear_op = lambda in_dim, out_dim, bias, name: nn.Linear(in_dim, out_dim, bias=bias)
        gate_linear_op = lambda in_dim, out_dim, bias, name: nn.Linear(in_dim, out_dim, bias=bias)
        attn_linear_op = lambda in_dim, out_dim, bias, name: nn.Linear(in_dim, out_dim, bias=bias)
        agg_linear_op = lambda in_dim, out_dim, bias, name: nn.Linear(in_dim, out_dim, bias=bias)
        clf_linear_op = lambda in_dim, out_dim, bias, name: nn.Linear(in_dim, out_dim, bias=bias)
        if patch_alpha is not None and patch_alpha is not None:
            if add_route:
                patch_linear_op = lambda in_dim, out_dim, bias, name: ProtoGuidedLoRA(in_dim, out_dim, bias=bias,
                                                                                      r=patch_rank,
                                                                                      lora_alpha=patch_alpha,
                                                                                      name=name)
            else:
                patch_linear_op = lambda in_dim, out_dim, bias, name: Linear(in_dim, out_dim, bias=bias, r=patch_rank,
                                                                             lora_alpha=patch_alpha)

        if classifier_alpha is not None and classifier_rank is not None:
            if add_route:
                clf_linear_op = lambda in_dim, out_dim, bias, name: ProtoGuidedLoRA(in_dim, out_dim, bias=bias,
                                                                                    r=classifier_rank,
                                                                                    lora_alpha=classifier_alpha,
                                                                                    name=name)
            else:
                clf_linear_op = lambda in_dim, out_dim, bias, name: Linear(in_dim, out_dim, bias=bias,
                                                                           r=classifier_rank,
                                                                           lora_alpha=classifier_alpha)
        if gate_rank is not None and gate_alpha is not None:
            if add_route:
                gate_linear_op = lambda in_dim, out_dim, bias, name: ProtoGuidedLoRA(in_dim, out_dim, bias=bias,
                                                                                     r=gate_rank,
                                                                                     lora_alpha=gate_alpha,
                                                                                     name=name)
            else:
                gate_linear_op = lambda in_dim, out_dim, bias, name: Linear(in_dim, out_dim, bias=bias, r=gate_rank, lora_alpha=gate_alpha)
        if agg_rank is not None and agg_alpha is not None:
            if add_route:
                agg_linear_op = lambda in_dim, out_dim, bias, name: ProtoGuidedLoRA(in_dim, out_dim, bias=bias,
                                                                                    r=agg_rank,
                                                                                    lora_alpha=agg_alpha,
                                                                                    name=name)
            else:
                agg_linear_op = lambda in_dim, out_dim, bias, name: Linear(in_dim, out_dim, bias=bias, r=agg_rank, lora_alpha=agg_alpha)
        if attn_rank is not None and attn_alpha is not None:
            if add_route:
                attn_linear_op = lambda in_dim, out_dim, bias, name: ProtoGuidedLoRA(in_dim, out_dim, bias=bias,
                                                                                      r=attn_rank,
                                                                                      lora_alpha=attn_alpha,
                                                                                      name=name)


            else:
                # lora
                attn_linear_op = lambda in_dim, out_dim, bias, name: Linear(in_dim, out_dim, bias=bias, r=attn_rank, lora_alpha=attn_alpha)


        # Renamed '_fc1' to 'patch_embed' for consistency
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[],
            dropout=dropout,
            out_dim=dim_hidden,
            end_with_fc=False,
            linear_op=patch_linear_op,
            prefix_name=f"{prefix_name}.patch_embed",
        )

        # self.gate_U = nn.Linear(dim_hidden, dim_hidden // 2)
        # self.gate_V = nn.Linear(dim_hidden, dim_hidden // 2)
        # self.gate_W = nn.Linear(dim_hidden // 2, dim_hidden)
        self.gate_U = gate_linear_op(dim_hidden, dim_hidden // 2, True, f"{prefix_name}.gate_U")
        self.gate_V = gate_linear_op(dim_hidden, dim_hidden // 2, True, f"{prefix_name}.gate_V")
        self.gate_W = gate_linear_op(dim_hidden // 2, dim_hidden, True, f"{prefix_name}.gate_W")

        # Attention mechanism layers

        self.W_head = attn_linear_op(dim_hidden, dim_hidden, True, f"{prefix_name}.W_head")
        self.W_tail = attn_linear_op(dim_hidden, dim_hidden, True, f"{prefix_name}.W_head")
        self.scale = dim_hidden ** -0.5

        # Aggregation layers
        if self.agg_type == 'gcn':
            self.linear = nn.Linear(dim_hidden, dim_hidden)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(dim_hidden * 2, dim_hidden)
        elif self.agg_type == 'bi-interaction':
            self.linear1 = agg_linear_op(dim_hidden, dim_hidden, True, f"{prefix_name}.linear1")
            self.linear2 = agg_linear_op(dim_hidden, dim_hidden, True, f"{prefix_name}.linear2")
        else:
            raise NotImplementedError(f"Aggregation type '{agg_type}' not supported.")

        self.activation = get_act(act)
        if self.dropout > 0:
            self.message_dropout = nn.Dropout(dropout)
        else:
            self.message_dropout = nn.Identity()

        # Pooling/Readout mechanism
        if self.pool == "mean":
            self.readout = global_mean_pool
        elif self.pool == "max":
            self.readout = global_max_pool
        elif self.pool == "attn":
            attn_net = nn.Sequential(
                # nn.Linear(dim_hidden, dim_hidden // 2),
                gate_linear_op(dim_hidden, dim_hidden // 2, True, f"{prefix_name}.readout.gate_nn.0"),  # 复用dim_hidden一半

                nn.LeakyReLU(),
                nn.Linear(dim_hidden // 2, 1)
            )
            self.readout = GeoGlobalAttention(attn_net)
        else:
            raise NotImplementedError(f"Pooling type '{self.pool}' not supported.")

        self.norm = nn.LayerNorm(dim_hidden)
        self.classifier = clf_linear_op(dim_hidden, self.num_classes, True, f"{prefix_name}.classifier")
        if self.enable_roupkt:
            # MoE
            self.top_k = min(self.n_experts, expert_topk)  # 3 by default
            # each expert is Identity or a simple MLP (FC + Act)
            self.experts = nn.ModuleList(
                [
                    create_mlp(
                        in_dim=embed_dim, hid_dims=[], dropout=dropout, out_dim=embed_dim,
                        end_with_fc=False
                    ) if expert_network == 'MLP' else nn.Identity()
                    for _ in range(self.n_experts)
                ]
            )
            self.router = nn.Linear(embed_dim, self.n_experts, bias=False)
            print(f"[DeepMIL_TFL_MoE] num_experts = {self.n_experts}; top_k = {self.top_k}.")

        self.initialize_weights()

    def forward_attention(self, h: torch.Tensor, attn_only: bool = False, **kwargs) -> tuple[
        torch.Tensor, torch.Tensor]:
        """
        Computes attention-based node embeddings and attention weights.

        Args:
            h (torch.Tensor): Input node features of shape (batch, nodes, features).
            attn_only (bool, optional): If True, only returns the attention matrix. Defaults to False.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            If attn_only is True:
                torch.Tensor: The full attention matrix after softmax, shape (batch, nodes, nodes).
            Else:
                Tuple[torch.Tensor, torch.Tensor]:
                    - Node embeddings after attention and aggregation, shape (batch, nodes, features).
                    - The full attention matrix after softmax, shape (batch, nodes, nodes).
        """
        h = self.patch_embed(h)
        h = (h + h.mean(dim=1, keepdim=True)) * 0.5

        e_h = self.W_head(h)
        e_t = self.W_tail(h)

        attn_logit = (e_h @ e_t.transpose(-2, -1)) * self.scale
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)

        # Create a full attention matrix for visualization/logging
        with torch.no_grad():
            full_attn = torch.full_like(attn_logit, float('-inf'))
            full_attn.scatter_(dim=-1, index=topk_index, src=topk_weight)
            full_attn = F.softmax(full_attn, dim=-1)

        batch_indices = torch.arange(e_t.size(0), device=h.device).view(-1, 1, 1)
        Nb_h = e_t[batch_indices, topk_index]

        topk_prob = F.softmax(topk_weight, dim=-1)
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h)

        gate = torch.tanh(e_h.unsqueeze(2).expand_as(Nb_h) + eh_r)
        ka_weight = torch.einsum('bnik,bnik->bni', gate, Nb_h)  # Element-wise product and sum
        ka_prob = F.softmax(ka_weight, dim=-1).unsqueeze(-1)
        e_Nh = torch.sum(ka_prob * Nb_h, dim=2)

        if self.agg_type == 'gcn':
            embedding = self.activation(self.linear(e_h + e_Nh))
        elif self.agg_type == 'sage':
            embedding = self.activation(self.linear(torch.cat([e_h, e_Nh], dim=-1)))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding

        if attn_only:
            return full_attn

        return embedding, full_attn

    def forward_features(self, h: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Extracts slide-level features from input tensor using attention-based aggregation.

        Args:
            h (torch.Tensor): Input features of shape (batch, nodes, features).
            attn_mask (torch.Tensor, optional): Optional attention mask (not used in this implementation).

        Returns:
            Tuple[torch.Tensor, Dict]:
                - h_norm: Normalized slide-level feature tensor.
                - dict: Dictionary containing raw attention weights under key 'attention'.
        """
        h, A_raw = self.forward_attention(h, attn_only=False)
        h = self.message_dropout(h)

        # Squeeze batch dimension for torch_geometric pooling functions
        h_pool = self.readout(h.squeeze(0), batch=None)

        h_norm = self.norm(h_pool)
        return h_norm, {'attention': A_raw}

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(h)
        return logits

    def forward(self, h: torch.Tensor,
                loss_fn: nn.Module = None,
                label: torch.LongTensor = None,
                attn_mask: torch.Tensor = None,
                return_attention: bool = False,
                return_slide_feats: bool = False,
                X_tfl=None
                ) -> torch.Tensor:
        """
        Forward pass for the WIKGMIL model.

        Args:
            h (torch.Tensor): Input features of shape (batch, nodes, features).
            loss_fn (nn.Module, optional): Loss function to compute classification loss.
            label (torch.LongTensor, optional): Ground truth labels.
            attn_mask (optional): Optional attention mask.
            return_attention (bool, optional): If True, return attention weights in log_dict.
            return_slide_feats (bool, optional): If True, return slide-level features in log_dict.

        Returns:
            Tuple[Dict, Dict]:
                - results_dict: Dictionary containing logits and loss.
                - log_dict: Dictionary containing attention weights, loss, and optionally slide features.
        """
        if self.enable_roupkt:
            h, log_dict = self.forward_features(h, attn_mask=attn_mask)
            router_logits = self.router(h)  # [B, num_experts]
            probs = F.softmax(router_logits, dim=-1)  # [B, num_experts]
            weights, selected_idxs = torch.topk(probs, self.top_k, dim=1)  # [B, top_k]

            expert_outputs = [expert(X_tfl[0, [i], :]) for i, expert in enumerate(self.experts)]
            stacked_expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, d, num_experts]
            moe_output = torch.sum(
                weights.unsqueeze(1) * stacked_expert_outputs[:, :, selected_idxs.squeeze()],
                dim=-1
            )  # [B, d]

            # logits = self.forward_head(wsi_feats, state_dict=state_dict)
            logits = self.forward_head(moe_output)

            if self.training:
                mask_1 = F.one_hot(selected_idxs.transpose(0, 1), self.n_experts).float()[0]  # (B, n_experts)
                density_1 = mask_1.mean(0)  # (n_experts, )
                density_1_proxy = probs.mean(
                    0)  # Something continuous that is correlated with what we want to equalize.

                balance_loss = (density_1_proxy * density_1).sum()
            else:
                balance_loss = 0

            if self.training:
                router_z_loss = torch.logsumexp(router_logits, dim=-1)
                router_z_loss = torch.square(router_z_loss)
                router_z_loss = router_z_loss.mean()
            else:
                router_z_loss = 0
            results_dict = {'logits': logits, 'loss': None, 'router_z_loss': router_z_loss,
                            'balance_loss': balance_loss}
            log_dict['loss'] = None
            if return_slide_feats:
                log_dict['slide_feats'] = moe_output
            return results_dict, log_dict
        else:
            h, log_dict = self.forward_features(h, attn_mask=attn_mask)
            logits = self.forward_head(h)

            cls_loss = self.compute_loss(loss_fn, logits, label)
            results_dict = {'logits': logits, 'loss': cls_loss}
            log_dict['loss'] = cls_loss.item() if cls_loss is not None else -1

            if not return_attention:
                del log_dict['attention']
            if return_slide_feats:
                log_dict['slide_feats'] = h

            return results_dict, log_dict

    def set_lora_enabled(self, enabled: bool = True):
        for module in self.modules():
            if isinstance(module, LoRALayer):
                module.set_lora_enabled(enabled)

    def set_lora_train_while_rest_is_mode(self, mode):
        self.train(mode)
        for module in self.modules():
            if isinstance(module, LoRALayer):
                if isinstance(module, nn.Module):
                    module.train(True)

# --- 3. PreTrainedModel Wrapper ---
class WIKGMILModel(PreTrainedModel):
    config_class = WIKGConfig

    def __init__(self, config: WIKGConfig, **kwargs):
        self.config = config
        super().__init__(config)
        for k, v in kwargs.items():
            setattr(config, k, v)

        self.model = WIKGMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_classes=config.num_classes,
            agg_type=config.agg_type,
            pool=config.pool,
            dropout=config.dropout,
            act=config.act,
            topk=config.topk,
            lora_alpha=config.lora_alpha,
            lora_rank=config.lora_rank,
            add_route=config.add_route,
            prefix_name="model",
            classifier_rank=config.classifier_rank,
            classifier_alpha=config.classifier_alpha,
            attn_rank=config.attn_rank,
            attn_alpha=config.attn_alpha,
            patch_rank=config.patch_rank,
            patch_alpha=config.patch_alpha,
            gate_rank=config.gate_rank,
            gate_alpha=config.gate_alpha,
            agg_rank=config.agg_rank,
            agg_alpha=config.agg_alpha,
            # ROUPKT
            enable_roupkt=config.enable_roupkt,
            expert_size=config.expert_size, expert_network=config.expert_network, expert_topk=config.expert_topk,

        )

        # Delegate forward methods to the internal model
        self.forward = self.model.forward
        self.forward_attention = self.model.forward_attention
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier


# --- 4. Register the model with AutoModel and AutoConfig ---
AutoConfig.register(WIKGConfig.model_type, WIKGConfig)
AutoModel.register(WIKGConfig, WIKGMILModel)


if __name__ == '__main__':
    features = torch.randn(1, 100, 512)
    # config = ABMILGatedBaseConfig(in_dim=512, embed_dim=256, attn_dim=384, lora_alpha=16, lora_rank=8)
    config = WIKGConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=4, topk=4,
                        agg_type="bi-interaction",
                        pool="attn", dropout=0.25, act="leaky_relu",
                        classifier_rank=2, classifier_alpha=0.5,
                        attn_rank=64, attn_alpha=16,
                        patch_rank=64,patch_alpha=16,
                        agg_rank=64, agg_alpha=16,
                        gate_rank=32, gate_alpha=8,
                        )
    model = WIKGMILModel(config)
    print(model)
    # model.model.set_lora_enabled(False)
    # model.model.set_lora_train_while_rest_is_mode(False)
    results_dict, log_dict = model(features,
                                   loss_fn=nn.CrossEntropyLoss(),
                                   label=torch.LongTensor([1]),
                                   return_attention=True,
                                   return_slide_feats=True
                                   )
    kvs = [(k, v)for k, v in model.state_dict().items()]
    F = 3