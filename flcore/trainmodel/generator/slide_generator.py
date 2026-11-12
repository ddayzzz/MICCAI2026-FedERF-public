import copy
from collections import defaultdict
from functools import partial

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from timm.layers import DropPath, set_fused_attn
from torch.nn import init


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop_v = attn_drop
        self.num_heads = num_heads
        self.proj_drop_v = proj_drop
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        assert qkv_bias == False, "qkv bias must be False"
        w_q = torch.empty((dim, dim), dtype=torch.float32)
        w_k = torch.empty((dim, dim), dtype=torch.float32)
        w_v = torch.empty((dim, dim), dtype=torch.float32)
        init.xavier_uniform_(w_q)
        init.xavier_uniform_(w_k,)
        init.xavier_uniform_(w_v, )
        self.q_proj = nn.Parameter(w_q)
        self.k_proj = nn.Parameter(w_k)
        self.v_proj = nn.Parameter(w_v)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, mask=None, **kwargs):
        B, C = x.shape
        Q = x @ self.q_proj
        K = x @ self.k_proj
        V = x @ self.v_proj
        Q = Q.reshape(B, self.num_heads, C // self.num_heads)  # [B, n_head, dim_of_head]
        K = K.reshape(B, self.num_heads, C // self.num_heads)
        V = V.reshape(B, self.num_heads, C // self.num_heads)



        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B, N_HEAD, dim_of_head, dim_of_head]
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)  # [B, n_head, n_head]
        attn = self.attn_drop(attn)


        x = (attn @ V).transpose(1, 2).reshape(B, C)  # attn @ V: [B, n_head, dim_of_head] ->
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )


    def forward(self, x, mask=None, **kwargs):
        attn = self.attn(self.norm1(x), mask=mask, **kwargs)
        x = x + self.drop_path(attn[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class WSIReconstructionTransformer(nn.Module):

    def __init__(self,
                 noise_dim,
                 input_dim,
                 input_embedding_dim,
                 num_heads,
                 out_dim,
            depth=12,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=None,
            ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.noise_dim = noise_dim
        self.input_dim = self.noise_dim + input_embedding_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, input_embedding_dim),
            nn.LeakyReLU(inplace=True),
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.input_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.proj = nn.Linear(self.input_dim, out_dim, bias=True)

    def forward(self, x, num_gen_slides):
        assert x.ndim == 2, "[b=1,n_dim]"
        assert x.size(0) == 1
        x = self.input_proj(x)
        x = x.expand(num_gen_slides, -1)
        eps = torch.rand((num_gen_slides, self.noise_dim))

        z = torch.cat((eps, x), dim=-1)

        for blk in self.blocks:
            z = blk(z)
        out = self.proj(z)
        return out


if __name__ == '__main__':
    model =  WSIReconstructionTransformer(noise_dim=64,  num_heads=4, depth=4, out_dim=512, input_embedding_dim=384, input_dim=384)
    x = torch.randn(1, 384)
    y = model(x, 10)
    f = 4