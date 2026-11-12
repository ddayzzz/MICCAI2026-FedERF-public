from typing import Literal

from flcore.trainmodel.lora_modules.lora_layers import Linear, LoRALayer
from flcore.trainmodel.mil_models.mil_template import MIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.trainmodel.mil_models.layers import GlobalAttention, GlobalGatedAttention, create_mlp
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel

from flcore.trainmodel.route_modules import ProtoGuidedLoRA

MODEL_TYPE = 'abmil'


class ABMIL(MIL):
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
            lora_rank: int = None,
            lora_alpha: int = None,
            add_route: bool = False,
            prefix_name: str = '',
            classifier_rank=None, classifier_alpha=None,
            attention_rank=None,attention_alpha=None,
            embedding_rank=None, embedding_alpha=None,
            final_attention_lora_multi=1,
            # ROUPKT
            enable_roupkt=False,
            expert_size=4,expert_network: Literal['Identity', 'MLP']="Identity", expert_topk=2,

    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.n_experts = expert_size
        self.expert_network = expert_network
        self.expert_topk = expert_topk
        self.enable_roupkt = enable_roupkt
        if classifier_rank is None:
            classifier_rank = lora_rank
            classifier_alpha = lora_alpha
        if attention_rank is None:
            attention_rank = lora_rank
            attention_alpha = lora_alpha
        if embedding_rank is None:
            embedding_rank = lora_rank
            embedding_alpha = lora_alpha
        classifier_mp = lambda in_dim, out_dim, bias, name: nn.Linear(in_dim, out_dim, bias=bias)
        attention_mp = lambda in_dim, out_dim, bias, name: nn.Linear(in_dim, out_dim, bias=bias)
        embedding_mp = lambda in_dim, out_dim, bias, name: nn.Linear(in_dim, out_dim, bias=bias)
        if classifier_rank is not None:
            classifier_mp = lambda in_dim, out_dim, bias, name: Linear(in_dim, out_dim, bias=bias, r=classifier_rank, lora_alpha=classifier_alpha)
            if add_route:
                assert classifier_rank is not None and classifier_alpha is not None and prefix_name is not None
                classifier_mp = lambda in_dim, out_dim, bias, name: ProtoGuidedLoRA(in_dim, out_dim, bias=bias, r=classifier_rank,
                                                                                lora_alpha=classifier_alpha,
                                                                                name=name)
        if attention_rank is not None:
            attention_mp = lambda in_dim, out_dim, bias, name: Linear(in_dim, out_dim, bias=bias, r=attention_rank,
                                                                       lora_alpha=attention_alpha)
            if add_route:
                assert attention_rank is not None and attention_alpha is not None and prefix_name is not None
                attention_mp = lambda in_dim, out_dim, bias, name: ProtoGuidedLoRA(in_dim, out_dim, bias=bias, r=attention_rank,
                                                                                lora_alpha=attention_alpha,
                                                                                name=name)
        if attention_rank is not None:
            embedding_mp = lambda in_dim, out_dim, bias, name: Linear(in_dim, out_dim, bias=bias, r=embedding_rank, lora_alpha=embedding_alpha)
            if add_route:
                assert embedding_rank is not None and embedding_alpha is not None and prefix_name is not None
                embedding_mp = lambda in_dim, out_dim, bias, name: ProtoGuidedLoRA(in_dim, out_dim, bias=bias, r=embedding_rank,
                                                                                lora_alpha=embedding_alpha,
                                                                                name=name)


        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] *
                     (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
            linear_op=embedding_mp,
            prefix_name=f"{prefix_name}.patch_embed",
        )

        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1,
            lora_rank=attention_rank, lora_alpha=attention_alpha, prefix_name=f"{prefix_name}.global_attn", add_route=add_route,
            final_attention_lora_multi=final_attention_lora_multi,
        )


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

        if num_classes > 0:
            # self.classifier = nn.Linear(embed_dim, num_classes)
            self.classifier = classifier_mp(embed_dim, num_classes, True, f"{prefix_name}.classifier")
        self.initialize_weights()

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
        if patched_emb is not None:
            h = patched_emb
        else:
            h = self.patch_embed(h)
        A = self.global_attn(h, state_dict=state_dict)  # B x M x K
        A = torch.transpose(A, -2, -1)  # B x K x M
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min

        if attn_only:
            return A
        return h, A

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = True, patched_emb=None, state_dict=None) -> torch.Tensor:
        """
        Compute bag-level features using attention pooling.

        Args:
            h (torch.Tensor): [B, M, D] input features.
            attn_mask (torch.Tensor, optional): Attention mask.

        Returns:
            Tuple[torch.Tensor, dict]: Bag features [B, D] and attention weights.
        """
        h, A_base = self.forward_attention(h, attn_mask=attn_mask, attn_only=False, patched_emb=patched_emb, state_dict=state_dict)  # A == B x K x M
        A = F.softmax(A_base, dim=-1)  # softmax over N
        h = torch.bmm(A, h).squeeze(dim=1)  # B x K x C --> B x C
        log_dict = {'attention': A_base if return_attention else None}
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
                state_dict=None,
                X_tfl= None,
                ) -> torch.Tensor:
        """
        Forward pass for ABMIL.

        Args:
            h: [B, M, D] input features.
            loss_fn: Optional loss function.
            label: Optional labels.
            attn_mask: Optional attention mask.
            X_tfl: [B x experts_feat x D]

        Returns:
            Tuple of (results_dict, log_dict) with logits and loss.
        """
        if self.enable_roupkt:
            wsi_feats, log_dict = self.forward_features(h, attn_mask=attn_mask, return_attention=return_attention,
                                                        patched_emb=patched_emb, state_dict=state_dict)
            router_logits = self.router(wsi_feats)  # [B, num_experts]
            probs = F.softmax(router_logits, dim=-1)  # [B, num_experts]
            weights, selected_idxs = torch.topk(probs, self.top_k, dim=1)  # [B, top_k]

            expert_outputs = [expert(X_tfl[0, [i], :]) for i, expert in enumerate(self.experts)]
            stacked_expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, d, num_experts]
            moe_output = torch.sum(
                weights.unsqueeze(1) * stacked_expert_outputs[:, :, selected_idxs.squeeze()],
                dim=-1
            )  # [B, d]

            # logits = self.forward_head(wsi_feats, state_dict=state_dict)
            logits = self.forward_head(moe_output, state_dict=state_dict)

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
            results_dict = {'logits': logits, 'loss': None, 'router_z_loss': router_z_loss, 'balance_loss': balance_loss}
            log_dict['loss'] = None
            if return_slide_feats:
                log_dict['slide_feats'] = moe_output
            return results_dict, log_dict
        else:
            wsi_feats, log_dict = self.forward_features(h, attn_mask=attn_mask, return_attention=return_attention,
                                                        patched_emb=patched_emb, state_dict=state_dict)
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

    def set_lora_eval_while_rest_is_mode(self, mode):
        self.train(mode)
        for module in self.modules():
            if isinstance(module, LoRALayer):
                if isinstance(module, nn.Module):
                    module.train(False)



class ABMILGatedBaseConfig(PretrainedConfig):
    """
    Configuration class for the ABMIL Gated Base model.

    This class stores the configuration parameters required to instantiate an ABMIL model
    with gated attention. It is compatible with Hugging Face's Transformers library and
    can be used to save, load, and share model configurations.

    Args:
        gate (bool): Whether to use gated attention (default: True).
        embed_dim (int): Embedding dimension after patch embedding (default: 512).
        attn_dim (int): Dimension of the attention mechanism (default: 384).
        num_fc_layers (int): Number of fully connected layers in the patch embedding MLP (default: 1).
        dropout (float): Dropout rate applied in the MLP and attention layers (default: 0.25).
        in_dim (int): Input feature dimension for each instance (default: 1024).
        num_classes (int): Number of output classes for the classification head (default: 2).
        **kwargs: Additional keyword arguments passed to the PretrainedConfig base class.

    Attributes:
        model_type (str): The model type identifier ("abmil").
        gate (bool): Whether to use gated attention.
        embed_dim (int): Embedding dimension after patch embedding.
        attn_dim (int): Dimension of the attention mechanism.
        num_fc_layers (int): Number of fully connected layers in the patch embedding MLP.
        dropout (float): Dropout rate applied in the MLP and attention layers.
        in_dim (int): Input feature dimension for each instance.
        num_classes (int): Number of output classes for the classification head.
        auto_map (dict): Mapping for Hugging Face AutoConfig and AutoModel registration.
    """

    model_type = MODEL_TYPE

    # add mapping

    # _target_: str = "src.models.abmil.ABMIL"
    def __init__(self,
                 gate: bool = True,
                 embed_dim: int = 512,
                 attn_dim: int = 384,
                 num_fc_layers: int = 1,
                 dropout: float = 0.25,
                 in_dim: int = 1024,
                 num_classes: int = 2,
                 lora_rank: int = None,
                 lora_alpha: int = None,
                 add_route: bool = False,
                 classifier_rank=None, classifier_alpha=None,
                 attention_rank=None, attention_alpha=None,
                 embedding_rank=None, embedding_alpha=None,
                 final_attention_lora_multi=1,
                 enable_roupkt=False,
                 expert_size=4, expert_network: Literal['Identity', 'MLP'] = "Identity", expert_topk=2,
                 **kwargs):
        super().__init__(**kwargs)
        self.gate = gate
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.num_fc_layers = num_fc_layers
        self.dropout = dropout
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.add_route = add_route
        self.classifier_rank = classifier_rank
        self.classifier_alpha = classifier_alpha
        self.attention_rank = attention_rank
        self.attention_alpha = attention_alpha
        self.embedding_rank = embedding_rank
        self.embedding_alpha = embedding_alpha
        self.final_attention_lora_multi = final_attention_lora_multi
        self.enable_roupkt = enable_roupkt
        self.expert_size = expert_size
        self.expert_network = expert_network
        self.expert_topk = expert_topk
        self.auto_map = {
            "AutoConfig": "modeling_abmil.ABMILGatedBaseConfig",
            "AutoModel": "modeling_abmil.ABMILModel",
        }


class ABMILModel(PreTrainedModel):
    config_class = ABMILGatedBaseConfig

    def __init__(self, config: ABMILGatedBaseConfig, **kwargs):
        """
        Initialize ABMILModel with the given config, allowing attribute overrides via kwargs.
        """

        self.config = config
        for k, v in kwargs.items():
            setattr(config, k, v)

        super().__init__(config)
        self.model = ABMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            attn_dim=config.attn_dim,
            gate=config.gate,
            num_classes=config.num_classes,
            lora_alpha=config.lora_alpha,
            lora_rank=config.lora_rank,
            add_route=config.add_route,
            prefix_name="model",
            classifier_rank=config.classifier_rank, classifier_alpha=config.classifier_alpha,
            attention_rank=config.attention_rank, attention_alpha=config.attention_alpha,
            embedding_rank=config.embedding_rank, embedding_alpha=config.embedding_alpha,
            final_attention_lora_multi=config.final_attention_lora_multi,
            enable_roupkt=config.enable_roupkt,
            expert_size=config.expert_size, expert_network=config.expert_network,
            expert_topk=config.expert_topk,

        )
        self.forward = self.model.forward
        self.forward_attention = self.model.forward_attention
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier


AutoConfig.register(ABMILGatedBaseConfig.model_type, ABMILGatedBaseConfig)
AutoModel.register(ABMILGatedBaseConfig, ABMILModel)


if __name__ == '__main__':
    features = torch.randn(1, 1000, 512).cuda(6)
    # config = ABMILGatedBaseConfig(in_dim=512, embed_dim=256, attn_dim=384, lora_alpha=16, lora_rank=8)
    config = ABMILGatedBaseConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=4,
                                  embedding_rank=64, embedding_alpha=128,
                                  attention_rank=16, attention_alpha=32,
                                  classifier_rank=None, classifier_alpha=None,
                                  final_attention_lora_multi=1,
                                  enable_roupkt=True,
                                  )
    model = ABMILModel(config).cuda(6)
    # model.model.set_lora_enabled(False)
    # model.model.set_lora_train_while_rest_is_mode(False)
    results_dict, log_dict = model(features,
                                   loss_fn=nn.CrossEntropyLoss(),
                                   label=torch.LongTensor([1]).cuda(6),
                                   return_attention=True,
                                   return_slide_feats=True,

                                   )
    kvs = [(k, v)for k, v in model.state_dict().items()]
    F = 3