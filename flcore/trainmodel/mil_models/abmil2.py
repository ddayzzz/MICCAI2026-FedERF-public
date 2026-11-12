from flcore.trainmodel.mil_models.abmil import ABMILGatedBaseConfig
from flcore.trainmodel.mil_models.mil_template import MIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.trainmodel.mil_models.layers import GlobalAttention, GlobalGatedAttention, create_mlp
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel

MODEL_TYPE = 'abmil'


class ABMIL2(MIL):
    """
    ABMIL (Attention-based Multiple Instance Learning) model.

    This class implements the core ABMIL architecture, which uses a patch embedding MLP,
    followed by a global attention or gated attention mechanism, and an optional classification head.

    Args:
        in_dim (int): Input feature dimension for each instance (default: 1024).
        embed_dim (int): Embedding dimension after patch embedding (default: 512).
        num_fc_layers (int): Number of fully connected layers in the patch embedding MLP (default: 1).
        dropout (float): Dropout rate applied in the MLP and attention layers (default: 0.25).
        attn_dim (int): Dimension of the attention mechanism (default: 384).
        gate (int): Whether to use gated attention (True) or standard attention (False) (default: True).
        num_classes (int): Number of output classes for the classification head (default: 2).
    """

    def __init__(
            self,
            in_dim: int = 1024,
            embed_dim: int = 512,
            num_fc_layers: int = 1,
            dropout: float = 0.25,
            attn_dim: int = 384,
            gate: int = True,
            num_classes: int = 2,
            num_anchors: int = 64,
            *kw,

    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] *
                     (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False
        )

        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1,
        )
        # 新增加的
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
        self.temperature = nn.Parameter(torch.ones(1))
        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)

        self.initialize_weights()
        nn.init.orthogonal_(self.anchors.data)  # 正交初始化是个好习惯

    def forward_attention(self, h: torch.Tensor, attn_mask=None, attn_only=True, patched_emb=None, state_dict=None) -> torch.Tensor:
        """
        Compute the attention scores (and optionally the embedded features) for the input instances.

        Args:
            h (torch.Tensor): Input tensor of shape [B, M, D], where B is the batch size,
                M is the number of instances (patches), and D is the input feature dimension.
            attn_mask (torch.Tensor, optional): Optional attention mask of shape [B, M], where 1 indicates
                valid positions and 0 indicates masked positions. If provided, masked positions are set to
                a very large negative value before softmax.
            attn_only (bool, optional): If True, return only the attention scores (A).
                If False, return a tuple (h, A) where h is the embedded features and A is the attention scores.
            patched_emb: encoded h

        Returns:
            torch.Tensor: If attn_only is True, returns the attention scores tensor of shape [B, K, M],
                where K is the number of attention heads (usually 1). If attn_only is False, returns a tuple
                (h, A) where h is the embedded features of shape [B, M, D'] and A is the attention scores.
        """

        h = self.patch_preprocessor(h)
        h_anchor_feat = self.anchor_map(self.anchors)  # num_anchor x D
        j_norm = F.normalize(h, dim=-1)  # B x M x D
        h_anchors_norm = F.normalize(h_anchor_feat, dim=-1)

        # 1. 计算相似度矩阵: [B, M, D]
        # 衡量每个 Patch 长得像哪个 Anchor
        similarity = torch.einsum('bmd, nd -> bmn', j_norm, h_anchors_norm)  # B x M x num_anchor
        assignment = F.softmax(similarity / self.temperature, dim=-1)  # [Batch, M, num_anchor]
        anchor_feat = assignment @ h_anchor_feat # Batch x M x D


        fused_feat = anchor_feat + h

        if patched_emb is not None:
            raise NotImplementedError
        else:
            h = self.patch_embed(fused_feat)


        A = self.global_attn(h, state_dict=state_dict)  # B x M x K
        A = torch.transpose(A, -2, -1)  # B x K x M
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min

        if attn_only:
            return A
        return h, A, h_anchor_feat, anchor_feat, assignment

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = True, patched_emb=None, state_dict=None) -> torch.Tensor:
        """
        Compute bag-level features using attention pooling.

        Args:
            h (torch.Tensor): [B, M, D] input features.
            attn_mask (torch.Tensor, optional): Attention mask.

        Returns:
            Tuple[torch.Tensor, dict]: Bag features [B, D] and attention weights.
        """
        h, A_base, h_anchor_feat, anchor_feat, anchor_sim = self.forward_attention(h, attn_mask=attn_mask, attn_only=False, patched_emb=patched_emb, state_dict=state_dict)  # A == B x K x M
        A = F.softmax(A_base, dim=-1)  # softmax over N
        h = torch.bmm(A, h).squeeze(dim=1)  # B x K x C --> B x C
        log_dict = {'attention': A_base if return_attention else None,
                    'anchor_feat': anchor_feat,
                    'h_anchor_feat': h_anchor_feat,
                    'anchor_sim': anchor_sim}
        return h, log_dict

    def forward_head(self, h: torch.Tensor, state_dict=None) -> torch.Tensor:
        """
        Args:
            h: [B x D]-dim torch.Tensor.

        Returns:
            logits: [B x num_classes]-dim torch.Tensor.
        """
        logits = self.classifier(h)
        return logits

    def forward(self, h: torch.Tensor,
                loss_fn: nn.Module = None,
                label: torch.LongTensor = None,
                attn_mask=None,
                return_attention: bool = False,
                return_slide_feats: bool = False,
                patched_emb=None,
                state_dict=None) -> torch.Tensor:
        """
        Forward pass for ABMIL.

        Args:
            h: [B, M, D] input features.
            loss_fn: Optional loss function.
            label: Optional labels.
            attn_mask: Optional attention mask.

        Returns:
            Tuple of (results_dict, log_dict) with logits and loss.
        """
        wsi_feats, log_dict = self.forward_features(h, attn_mask=attn_mask, return_attention=return_attention, patched_emb=patched_emb, state_dict=state_dict)
        logits = self.forward_head(wsi_feats, state_dict=state_dict)
        cls_loss = MIL.compute_loss(loss_fn, logits, label)
        results_dict = {'logits': logits, 'loss': cls_loss}
        log_dict['loss'] = cls_loss.item() if cls_loss is not None else -1
        if return_slide_feats:
            log_dict['slide_feats'] = wsi_feats
        return results_dict, log_dict

    def forward_encode_features(self, h: torch.Tensor, state_dict=None):
        h = self.patch_embed(h)
        return h



class ABMILModel2(PreTrainedModel):
    config_class = ABMILGatedBaseConfig

    def __init__(self, config: ABMILGatedBaseConfig, **kwargs):
        """
        Initialize ABMILModel with the given config, allowing attribute overrides via kwargs.
        """

        self.config = config
        for k, v in kwargs.items():
            setattr(config, k, v)

        super().__init__(config)
        self.model = ABMIL2(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            attn_dim=config.attn_dim,
            gate=config.gate,
            num_classes=config.num_classes,
        )
        self.forward = self.model.forward
        self.forward_attention = self.model.forward_attention
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier



if __name__ == '__main__':
    g = torch.tensor([2,4,5,1,88,7])
    b = torch.topk(g, 2, )
    patch_embeddings = torch.zeros((50, 768))
    cluster_embeddings = torch.zeros((60, 768))
    a = -torch.cdist(patch_embeddings, cluster_embeddings)


    features = torch.randn(1, 100, 512)
    config = ABMILGatedBaseConfig(in_dim=512, embed_dim=256, attn_dim=384, lora_alpha=16, lora_rank=8)
    model = ABMILModel2(config)
    results_dict, log_dict = model(features,
                                   loss_fn=nn.CrossEntropyLoss(),
                                   label=torch.LongTensor([1]),
                                   return_attention=True,
                                   return_slide_feats=True
                                   )
    F = 3