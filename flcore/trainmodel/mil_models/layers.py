import torch.nn as nn
import torch
import torch.nn.functional as F
from flcore.trainmodel.lora_modules.lora_layers import Linear as LoraLinear, FunctionalLinear
from flcore.trainmodel.route_modules import ProtoGuidedLoRA


def create_mlp(
        in_dim=768,
        hid_dims=[512, 512],
        out_dim=512,
        act=nn.ReLU(),
        dropout=0.,
        end_with_fc=True,
        end_with_dropout=False,
        bias=True,
        linear_op=None,
        prefix_name=""
):
    if linear_op is not None:
        assert prefix_name is not None
    layers = []
    if len(hid_dims) < 0:
        mlp = nn.Identity()
    elif len(hid_dims) >= 0:
        layer_id = 0
        if len(hid_dims) > 0:
            for li, hid_dim in enumerate(hid_dims):
                if linear_op is not None:
                    layers.append(linear_op(in_dim, hid_dim, bias, f"{prefix_name}.{li}"))
                else:
                    layers.append(nn.Linear(in_dim, hid_dim, bias=bias))
                layers.append(act)
                layers.append(nn.Dropout(dropout))
                in_dim = hid_dim
                layer_id += 1
        #
        if linear_op is not None:
            layers.append(linear_op(in_dim, out_dim, bias, f"{prefix_name}.{layer_id}"))
        else:
            layers.append(nn.Linear(in_dim, out_dim))
        if not end_with_fc:
            layers.append(act)
        if end_with_dropout:
            layers.append(nn.Dropout(dropout))
        mlp = nn.Sequential(*layers)
    return mlp


#
# Attention networks
#
class GlobalAttention(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        num_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=0., num_classes=1):
        super().__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(D, num_classes)]

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x)  # N x num_classes


class GlobalGatedAttention(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        num_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=0., num_classes=1, lora_rank=None, lora_alpha=None, add_route=False, prefix_name=None, final_attention_lora_multi=1):
        super().__init__()
        self.prefix_name = prefix_name
        linear_mp = lambda in_dim, out_dim, name: nn.Linear(in_dim, out_dim)
        final_mp = lambda in_dim, out_dim, name: nn.Linear(in_dim, out_dim)
        if lora_rank is not None:
            linear_mp = lambda in_dim, out_dim, name: LoraLinear(in_dim, out_dim, r=lora_rank, lora_alpha=lora_alpha)
            # final_mp = lambda in_dim, out_dim, name: LoraLinear(in_dim, out_dim, r=int(lora_rank * final_attention_lora_multi), lora_alpha=int(lora_alpha * final_attention_lora_multi))
        if add_route:
            assert lora_rank is not None and lora_alpha is not None and prefix_name is not None
            linear_mp = lambda in_dim, out_dim, name: ProtoGuidedLoRA(in_dim, out_dim, r=lora_rank, lora_alpha=lora_alpha, name=f"{prefix_name}.{name}")
            # final_mp = lambda in_dim, out_dim, name: ProtoGuidedLoRA(in_dim, out_dim, r=int(lora_rank * final_attention_lora_multi), lora_alpha=int(lora_alpha * final_attention_lora_multi), name=f"{prefix_name}.{name}")
        self.attention_a = [
            linear_mp(L, D, name=f"attention_a.0"),
            nn.Tanh(),
            nn.Dropout(dropout)
        ]

        self.attention_b = [
            linear_mp(L, D, name="attention_b.0"),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        ]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = final_mp(D, num_classes, name="attention_c")  # 不要lora!

    def forward(self, x, state_dict=None):
        if state_dict is not None:
            a = self.attention_a[0](x, lora_A=state_dict[f"{self.prefix_name}.attention_a.0.lora_A"], lora_B=state_dict[f"{self.prefix_name}.attention_a.0.lora_B"])
            for m in list(self.attention_a.children())[1:]:
                a = m(a)
            b = self.attention_a[0](x, lora_A=state_dict[f"{self.prefix_name}.attention_b.0.lora_A"],
                                    lora_B=state_dict[f"{self.prefix_name}.attention_b.0.lora_B"])
            for m in list(self.attention_b.children())[1:]:
                b = m(b)
            A = a.mul(b)
            A = self.attention_c(A, lora_A=state_dict[f"{self.prefix_name}.attention_c.lora_A"],
                                    lora_B=state_dict[f"{self.prefix_name}.attention_c.lora_B"])
        else:
            a = self.attention_a(x)
            b = self.attention_b(x)

            A = a.mul(b)
            A = self.attention_c(A)  # N x num_classes
        return A

# patch编码器对象
class AnchorEmbedding(nn.Module):
    def __init__(self, num_anchors, in_dim,

                 initialize_anchor_weights: torch.Tensor = None,
                 similarity_method="l2",
                 hard_assignment=False):
        super().__init__()
        self.hard_assignment = hard_assignment
        if initialize_anchor_weights is not None:
            self.anchors = nn.Parameter(initialize_anchor_weights.data.clone(), requires_grad=True)
        else:
            self.anchors = nn.Parameter(torch.randn(num_anchors, in_dim), requires_grad=True)
        self.anchor_map = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(inplace=True)
        )
        self.patch_preprocessor = nn.Sequential(
            nn.Linear(in_dim, in_dim),  # 保持abmil的patch embedding输入不变
            nn.LeakyReLU(inplace=True)
        )
        # self.norm_layer = nn.LayerNorm(in_dim)
        self.similarity_scale = nn.Parameter(torch.ones(1))
        self.similarity_method = similarity_method

    def _straight_through_softmax(self, logits, hard_assignment=True, dim=-1):
        y_soft = F.softmax(logits / self.similarity_scale, dim=1)
        if hard_assignment:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

    def _get_contextual_features(self, patch_embeddings, cluster_embeddings):
        """
        计算图块和聚类之间的相似度，并生成上下文感知的特征。

        Args:
            patch_embeddings (Tensor): 图块嵌入, 形状 (num_patches, embedding_dim).
            cluster_embeddings (Tensor): 聚类嵌入, 形状 (num_clusters, embedding_dim).

        Returns:
            Tensor: 上下文特征，通过加权聚类中心得到, 形状 (num_patches, embedding_dim).
        """
        if self.similarity_method == 'l2':
            # 基于L2距离计算相似度（负距离）
            similarity_scores = -torch.cdist(patch_embeddings, cluster_embeddings)  # [num_patches, num_clusters]
        else:  # 默认为点积相似度
            similarity_scores = patch_embeddings @ cluster_embeddings.transpose(-2, -1)

        # 获得图块到聚类的分配权重
        assignment_weights = self._straight_through_softmax(similarity_scores, self.hard_assignment, dim=1) # [num_patches, num_clusters]

        # 根据分配权重对聚类中心进行加权求和，得到每个图块的上下文特征
        contextual_features = torch.matmul(assignment_weights, cluster_embeddings)  # [num_patches, embedding_dim]
        return contextual_features

    def forward(self, h):
        h = h.squeeze()
        h_patches = self.patch_preprocessor(h)  # [M, D]
        h_anchor = self.anchor_map(self.anchors)  # num_anchor x D
        contextual_features = self._get_contextual_features(h_patches, h_anchor)
        # 融合后的anchor特征
        patch_embeddings = h_patches + contextual_features
        # 计算路由LoRA的路由特征



