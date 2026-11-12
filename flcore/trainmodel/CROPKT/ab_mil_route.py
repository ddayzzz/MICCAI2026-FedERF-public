import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flcore.trainmodel.mil_models.layers import create_mlp, GlobalGatedAttention, GlobalAttention


# from model.layers import *
# from utils.func import gumbel_noise




class DeepMIL_TFL_MoE(nn.Module):
    """
    Deep Multiple Instance Learning with Transfer Learning (TFL) features.

    Args:
        dim_in: input instance dimension.
        dim_emb: instance embedding dimension.
        num_cls: the number of class to predict.
        pooling: the type of MIL pooling, one of 'mean', 'max', and 'attention', default by attention pooling.
    """

    def __init__(self, dim_in=768, dim_emb=512, num_cls=2, dim_attn=384, num_feat_proj_layers=1, drop_rate=0.25,
                 pooling='attention', pred_head='default', expert_size=13, expert_network='Identity', expert_topk=5,
                 noise_gates=False, **kwargs):
        super().__init__()
        assert pooling in ['mean', 'max', 'attention', 'gated_attention']
        assert pred_head in ['default']
        assert expert_network in ['Identity', 'MLP']
        self.n_experts = expert_size
        self.noise_gates = noise_gates
        self.noise_mult = 1.

        # MIL network employs as Router
        self.feat_proj = create_mlp(
            in_dim=dim_in,
            hid_dims=[dim_emb] * (num_feat_proj_layers - 1),
            dropout=drop_rate,
            out_dim=dim_emb,
            end_with_fc=False
        )

        self.agg_method = pooling
        # if pooling == 'gated_attention':
        #     self.attention_net = GlobalGatedAttention(L=dim_emb, D=dim_attn, dropout=drop_rate)
        # elif pooling == 'attention':
        #     self.attention_net = Attn_Net(L=dim_emb, D=dim_attn, dropout=drop_rate)
        # else:
        #     self.attention_net = None
        if pooling == "gated_attention":
            attn_func = GlobalGatedAttention
        elif pooling == "":
            attn_func = GlobalAttention
        else:
            raise NotImplementedError("Not Implemented!")
        self.attention_net = attn_func(
            L=dim_emb,
            D=dim_attn,
            dropout=drop_rate,
            num_classes=1
        )

        self.router = nn.Linear(dim_emb, self.n_experts, bias=False)

        # MoE
        self.top_k = min(self.n_experts, expert_topk)  # 3 by default
        # each expert is Identity or a simple MLP (FC + Act)
        self.experts = nn.ModuleList(
            [
                create_mlp(
                    in_dim=dim_emb, hid_dims=[], dropout=drop_rate, out_dim=dim_emb,
                    end_with_fc=False
                ) if expert_network == 'MLP' else nn.Identity()
                for _ in range(self.n_experts)
            ]
        )
        print(f"[DeepMIL_TFL_MoE] num_experts = {self.n_experts}; top_k = {self.top_k}.")

        self.pred_head = nn.Linear(dim_emb, num_cls)

        self.register_buffer('zero', torch.zeros((1,)), persistent=False)

    def forward_attention_pooling(self, X, attn_mask=None):
        # X is B x K x C (K is the number of instances)
        # num_head = 1 for ABMIL
        A = self.attention_net(X)  # B x K x num_head
        A = torch.transpose(A, -2, -1)  # B x num_head x K

        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min
        A = F.softmax(A, dim=-1)  # softmax over K (the last dim)
        M = torch.bmm(A, X).squeeze(dim=1)  # B x num_head x C --> B x C

        return M, A.squeeze(dim=1)  # B x C, B x K

    def forward_slide_representation(self, X):
        assert X.shape[0] == 1
        X = self.feat_proj(X)

        # global pooling: B x K x C -> B x C
        if 'attention' in self.agg_method:
            out_feat, attn = self.forward_attention_pooling(X)
            return out_feat, attn
        elif self.agg_method == 'mean':
            out_feat = torch.mean(X, dim=1)
            return out_feat
        elif self.agg_method == 'max':
            out_feat, _ = torch.max(X, dim=1)
            return out_feat
        else:
            raise NotImplementedError("Not Implemented!")

    def forward(self, X_tfl, X, ret_with_attn=False, ret_bag_feat=False):
        """
        X_tfl: bag-level features output by transfer learning, B x K' x d.
        X: initial bag features, with shape B x K x C
           where B = 1 for batch size, K is the instance size of this bag, and C is feature dimension.
        """
        assert self.n_experts == X_tfl.shape[1]

        # pass forward a MIL-based router
        slide_results = self.forward_slide_representation(X)
        if isinstance(slide_results, tuple):
            out_feat, attn = slide_results
        else:
            out_feat = slide_results
        router_logits = self.router(out_feat)  # [B, num_experts]

        # add noise into routing scores:
        # refer to https://github.com/lucidrains/st-moe-pytorch/blob/main/st_moe_pytorch/st_moe_pytorch.py#L424
        maybe_noised_gate_logits = router_logits
        if self.noise_gates and self.training:
            noise = gumbel_noise(maybe_noised_gate_logits)
            maybe_noised_gate_logits = maybe_noised_gate_logits + noise * self.noise_mult

        # pass forward the TopK experts and aggregate their represetation outputs
        probs = F.softmax(maybe_noised_gate_logits, dim=-1)  # [B, num_experts]
        weights, selected_idxs = torch.topk(probs, self.top_k, dim=1)  # [B, top_k]

        expert_outputs = [expert(X_tfl[0, [i], :]) for i, expert in enumerate(self.experts)]
        stacked_expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, d, num_experts]
        moe_output = torch.sum(
            weights.unsqueeze(1) * stacked_expert_outputs[:, :, selected_idxs.squeeze()],
            dim=-1
        )  # [B, d]

        logit = self.pred_head(moe_output)  # B x num_cls

        # balance losses - (batch, experts)
        # We want to equalize the fraction of the batch assigned to each expert
        # refer to https://github.com/lucidrains/st-moe-pytorch/blob/main/st_moe_pytorch/st_moe_pytorch.py
        if self.training:
            mask_1 = F.one_hot(selected_idxs.transpose(0, 1), self.n_experts).float()[0]  # (B, n_experts)
            density_1 = mask_1.mean(0)  # (n_experts, )
            density_1_proxy = probs.mean(0)  # Something continuous that is correlated with what we want to equalize.

            balance_loss = (density_1_proxy * density_1).sum()
        else:
            balance_loss = self.zero

        # calculate the router z-loss proposed in paper
        # Router z-loss is an auxiliary loss function designed to improve the stability and performance of
        # mixture of experts (MoE) models by penalizing large logits during training.
        # refer to https://github.com/lucidrains/st-moe-pytorch/blob/main/st_moe_pytorch/st_moe_pytorch.py
        if self.training:
            router_z_loss = torch.logsumexp(router_logits, dim=-1)
            router_z_loss = torch.square(router_z_loss)
            router_z_loss = router_z_loss.mean()
        else:
            router_z_loss = self.zero

        if not self.training and ret_bag_feat:
            return logit, moe_output.detach()  # B x num_cls, B x d

        if not self.training and ret_with_attn:
            return logit, probs.detach()  # B x num_cls, B x num_experts

        if self.training:
            return logit, probs.detach(), balance_loss, router_z_loss
        else:
            return logit
