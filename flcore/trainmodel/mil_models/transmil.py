import torch
import torch.nn as nn
import numpy as np
from flcore.trainmodel.mil_models.mil_template import MIL
from flcore.trainmodel.lora_modules import lora_layers
# from nystrom_attention import NystromAttention
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig
# from src.models.mil_template import MIL
from flcore.trainmodel.mil_models.layers import GlobalAttention, GlobalGatedAttention, create_mlp
from flcore.trainmodel.route_modules import ProtoGuidedLoRA, MoEConv2DLoRA, MoEConvNDLoRA
from flcore.trainmodel.trans_mil.nystrom_attention import NystromAttention
from typing import Literal
import torch.nn.functional as F


class TransLayer(nn.Module):
    def __init__(self, norm_layer: nn.Module = nn.LayerNorm, dim: int = 512, num_heads: int = 8):
        """
        Transformer Layer with Nystrom Attention.

        Args:
            norm_layer (nn.Module): Normalization layer, default is nn.LayerNorm.
            dim (int): Dimension for the transformer layer, default is 512.
        """
        super().__init__()
        self.norm = norm_layer(dim)
        self.attention = NystromAttention(
            dim=dim,
            dim_head=dim // num_heads,
            heads=num_heads,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the transformer layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention and normalization.
        """
        x = x + self.attention(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim: int = 512, prefix_name: str = None, proj_rank=None, proj_alpha=None, add_route=False):
        """
        Position-wise Projection Embedded Gradient (PPEG) for positional encoding.

        Args:
            dim (int): Dimension for the embedding, default is 512.
        """
        super(PPEG, self).__init__()
        if proj_rank is not None and proj_alpha is not None:
            if add_route:
                proj_conv = lambda x, kn, stride, padding, gp, name: MoEConv2DLoRA(x, x, kn, name=name, stride=stride,
                                                                                   r=proj_rank,
                                                                                   lora_alpha=proj_alpha,
                                                                                   groups=dim, padding=padding,
                                                                                   )
            else:
                proj_conv = lambda x, kn, stride, padding, gp, name: lora_layers.Conv2d(x, x, kn, stride=stride, r=proj_rank,
                                                                               lora_alpha=proj_alpha,
                                                                               groups=dim, padding=padding)
        else:
            proj_conv = lambda x, kn, stride, padding, gp, name: nn.Conv2d(x, x, kn, stride=stride, groups=dim,padding=padding)
        # self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        # self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        # self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)
        self.proj = proj_conv(dim, 7, 1, 7 // 2, dim, f"{prefix_name}.proj")
        self.proj1 = proj_conv(dim, 5, 1, 5 // 2, dim, f"{prefix_name}.proj1")
        self.proj2 = proj_conv(dim, 3, 1, 3 // 2, dim, f"{prefix_name}.proj2")

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Forward pass for the PPEG layer.

        Args:
            x (torch.Tensor): Input tensor.
            H (int): Height for reshaping.
            W (int): Width for reshaping.

        Returns:
            torch.Tensor: Output tensor with positional encoding applied.
        """
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(MIL):
    def __init__(self, in_dim: int, embed_dim: int,
                 num_fc_layers: int, dropout: float,
                 num_attention_layers: int, num_classes: int, num_heads: int=8,
                 add_route=False,prefix_name=None,
                 patch_rank=None, patch_alpha=None,
                 ppeg_proj_rank=None, ppeg_proj_alpha=None,
                 classifier_rank=None, classifier_alpha=None,
                 # ROUPKT
                 enable_roupkt=False,
                 expert_size=4, expert_network: Literal['Identity', 'MLP'] = "Identity", expert_topk=2,

                 ):
        """
        TransMIL model with transformer-based Multi-instance Learning.

        Args:
            in_dim (int): Input dimension for the MLP.
            embed_dim (int): Embedding dimension for all layers.
            n_fc_layers (int): Number of fully connected layers in the MLP.
            dropout (float): Dropout rate for MLP.
            n_attention_layers (int): Number of transformer attention layers.
            n_classes (int): Number of output classes for classification.
        """
        super(TransMIL, self).__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.n_experts = expert_size
        self.expert_network = expert_network
        self.expert_topk = expert_topk
        self.enable_roupkt = enable_roupkt
        patch_linear_op = lambda in_dim, out_dim, bias, name: nn.Linear(in_dim, out_dim, bias=bias)
        clf_linear_op = lambda in_dim, out_dim, bias, name: nn.Linear(in_dim, out_dim, bias=bias)
        if patch_alpha is not None and patch_alpha is not None:
            if add_route:
                patch_linear_op = lambda in_dim, out_dim, bias, name: ProtoGuidedLoRA(in_dim, out_dim, bias=bias,
                                                                                      r=patch_rank,
                                                                                      lora_alpha=patch_alpha,
                                                                                      name=name)
            else:
                patch_linear_op = lambda in_dim, out_dim, bias, name: lora_layers.Linear(in_dim, out_dim, bias=bias, r=patch_rank,
                                                                             lora_alpha=patch_alpha)
        if classifier_rank is not None and classifier_rank is not None:
            if add_route:
                clf_linear_op = lambda in_dim, out_dim, bias, name: ProtoGuidedLoRA(in_dim, out_dim, bias=bias,
                                                                                      r=classifier_rank,
                                                                                      lora_alpha=classifier_alpha,
                                                                                      name=name)
            else:
                clf_linear_op = lambda in_dim, out_dim, bias, name: lora_layers.Linear(in_dim, out_dim, bias=bias,
                                                                                         r=classifier_rank,
                                                                                         lora_alpha=classifier_alpha)
        self.patch_embed: nn.Module = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
            linear_op=patch_linear_op,
            prefix_name=f"{prefix_name}.patch_embed",
        )

        self.pos_layer: nn.Module = PPEG(dim=embed_dim, prefix_name=f"{prefix_name}.pos_layer",
                                         proj_alpha=ppeg_proj_alpha, proj_rank=ppeg_proj_rank, add_route=add_route)
        self.cls_token: nn.Parameter = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.blocks: nn.ModuleList = nn.ModuleList(
            [TransLayer(dim=embed_dim, num_heads=num_heads) for _ in range(num_attention_layers)]
        )

        self.norm: nn.LayerNorm = nn.LayerNorm(embed_dim)
        # self.classifier: nn.Linear = nn.Linear(embed_dim, num_classes)
        self.classifier: nn.Linear = clf_linear_op(embed_dim, num_classes, True, f"{prefix_name}.classifier")
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

    def forward_attention(self, h: torch.Tensor) -> torch.Tensor:
        pass

    def forward_features(self, h: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Get slide-level features from cls token.

        Args:
            h (torch.Tensor): The input tensor of shape (features, dim) or
                              (batch_size, features, dim).

        Returns:
            torch.Tensor: Slide-level feature of cls token. Output shape will be
                          (1, 1, embed_dim) if input was 2D or
                          (batch_size, 1, embed_dim) if input was 3D.
        """
        if len(h.shape) == 2:
            h = h.unsqueeze(0)
        h = self.patch_embed(h)
        h, h_square, w_square = self._square_pad(h)
        h = self._add_cls_token(h)
        h, attn = self._apply_trans_layers(h, h_square, w_square, return_attention)
        wsi_feat = self.norm(h)[:, 0]  # get cls token
        return wsi_feat, {'attention': attn}

    def _apply_trans_layers(self, h: torch.Tensor, h_square: int, w_square: int, return_attention: bool = False) -> torch.Tensor:
        """
        Apply transformer layers to the input.

        Args:
            h (torch.Tensor): Input tensor after adding class token.
            h_square (int): Square height obtained from padding calculation.
            w_square (int): Square width obtained from padding calculation.
            return_attention (bool): whether to compute attention scores wrt cls token

        Returns:
            torch.Tensor: Transformed tensor.
        """
        intermed_dict = {}
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == 0:
                if return_attention:
                    # compute attention scores wrt cls token in first position
                    cls_token = h[:, 0]  # b x d
                    feats = h[:, 1:] # b x n x d
                    # compute the dot prod similarity between each feat and cls token
                    intermed_dict['attention'] = torch.matmul(feats, cls_token.unsqueeze(-1)).squeeze(-1)
                h = self.pos_layer(h, h_square, w_square)
        return h, intermed_dict

    def _square_pad(self, h: torch.Tensor) -> tuple:
        """
        Pad feature tensor to make it square.

        Args:
            h (torch.Tensor): Input tensor.

        Returns:
            tuple: Padded tensor, square height, and square width.
        """
        H = h.shape[1]
        add_length, h_square, w_square = self._get_square_length(H)
        h = torch.cat([h, h[:, :add_length, :]], dim=1)
        return h, h_square, w_square

    def _add_cls_token(self, h: torch.Tensor) -> torch.Tensor:
        """
        Add class token to the input tensor.

        Args:
            h (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Input tensor with class token added.
        """
        B, H = h.shape[0], h.shape[1]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        return h

    def _get_square_length(self, H: int) -> tuple:
        """
        Calculate the required lengths to convert the input into square form.

        Args:
            H (int): Original height (or length) of the input tensor.

        Returns:
            tuple: Additional length needed, and new square dimensions.
        """
        h_square, w_square = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = h_square * w_square - H
        return add_length, h_square, w_square

    def forward_head(self, wsi_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head.

        Args:
            wsi_feat (torch.Tensor): Slide-level feature for classification.

        Returns:
            torch.Tensor: Logits for classification.
        """
        logits = self.classifier(wsi_feat)
        return logits

    def forward(self, h: torch.Tensor,
                loss_fn: nn.Module=None,
                label: torch.LongTensor=None,
                attn_mask=None,
                return_attention: bool = False,
                return_slide_feats: bool = False,
                X_tfl=None) -> tuple:
        """
        Complete forward pass of the model.

        Args:
            h (torch.Tensor): Input feature tensor.

        Returns:
            tuple: Slide-level features and logits from the classifier.
        """
        if self.enable_roupkt:
            wsi_feats, log_dict = self.forward_features(h, return_attention=return_attention)
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
            wsi_feats, log_dict = self.forward_features(h, return_attention=return_attention)
            logits = self.forward_head(wsi_feats)
            cls_loss = self.compute_loss(loss_fn=loss_fn, label=label, logits=logits)
            results_dict = {'logits': logits, 'loss': cls_loss}
            log_dict = {'loss': cls_loss.item() if cls_loss is not None else -1}
            if return_slide_feats:
                log_dict['slide_feats'] = wsi_feats
            return results_dict, log_dict

    def set_lora_enabled(self, enabled: bool = True):
        for module in self.modules():
            if isinstance(module, lora_layers.LoRALayer):
                module.set_lora_enabled(enabled)

    def set_lora_train_while_rest_is_mode(self, mode):
        self.train(mode)
        for module in self.modules():
            if isinstance(module, lora_layers.LoRALayer):
                if isinstance(module, nn.Module):
                    module.train(True)

#@dataclass
class TransMILConfig(PretrainedConfig):
    model_type = 'transmil'

    def __init__(self,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        dropout: float = 0.25,
        in_dim: int = 1024,
        num_classes: int = 2,
        num_attention_layers: int = 2,
        num_heads: int = 4,
        add_route=False,
        patch_rank=None, patch_alpha=None,
        ppeg_proj_rank=None, ppeg_proj_alpha=None,
        classifier_rank=None, classifier_alpha=None,
        enable_roupkt=False,
        expert_size=4, expert_network: Literal['Identity', 'MLP'] = "Identity", expert_topk=2,
        **kwargs
        ):
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_fc_layers = num_fc_layers
        self.dropout = dropout
        self.num_attention_layers = num_attention_layers
        self.num_heads = num_heads
        self.add_route = add_route
        self.patch_rank = patch_rank
        self.patch_alpha = patch_alpha
        self.ppeg_proj_rank = ppeg_proj_rank
        self.ppeg_proj_alpha = ppeg_proj_alpha
        self.classifier_rank = classifier_rank
        self.classifier_alpha = classifier_alpha
        self.enable_roupkt = enable_roupkt
        self.expert_size = expert_size
        self.expert_network = expert_network
        self.expert_topk = expert_topk
        super().__init__(**kwargs)


class TransMILModel(PreTrainedModel):
    config_class = TransMILConfig

    def __init__(self, config: TransMILConfig, **kwargs):
        """
        load a model with the given config. Overwrite config attributes with any model kwargs
        """
        self.config = config
        super().__init__(config)
        for k,v in kwargs.items():
            setattr(config, k, v)

        self.model = TransMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_classes=config.num_classes,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            num_heads=config.num_heads,
            num_attention_layers=config.num_attention_layers,
            add_route=config.add_route, prefix_name="model",
            patch_rank=config.patch_rank, patch_alpha=config.patch_alpha,
            ppeg_proj_rank=config.ppeg_proj_rank, ppeg_proj_alpha=config.ppeg_proj_alpha,
            classifier_rank=config.classifier_rank, classifier_alpha=config.classifier_alpha,
            enable_roupkt=config.enable_roupkt,expert_size=config.expert_size,expert_network=config.expert_network,
            expert_topk=config.expert_topk,
        )
        self.forward = self.model.forward
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier



AutoConfig.register("transmil", TransMILConfig)
AutoModel.register(TransMILConfig, TransMILModel)

if __name__ == '__main__':
    features = torch.randn(1, 100, 512)
    # config = ABMILGatedBaseConfig(in_dim=512, embed_dim=256, attn_dim=384, lora_alpha=16, lora_rank=8)
    config = TransMILConfig(in_dim=512, embed_dim=128, attn_dim=32, num_classes=4,
                            num_heads=4, num_fc_layers=1, num_attention_layers=2,
                            patch_rank=64, patch_alpha=16,
                            ppeg_proj_rank=0.5, ppeg_proj_alpha=1,
                            classifier_rank=2, classifier_alpha=0.5,
                            add_route=True

                            )
    pk = "/xxx/projects2/fedmed2026/models/data_tcga_model_trans_mil2_rankMidFeat_alphax0.25_lora/eval_site_LIHC/Our_glb30_loc1_lr0.0002_wd0.00001_optadamw_nc4jr1.0_bs1_noLRdecay_0209_loraLast_finetune30Epc_server_fold0.pt"

    model = TransMILModel(config)

    lora_state_dicts = []
    ck = torch.load(pk, map_location=torch.device('cpu'))

    for client_name, sub_state_dict in ck['client_models'].items():
        lora_state_dicts.append(sub_state_dict)
    for module in model.modules():
        if isinstance(module, (ProtoGuidedLoRA, MoEConvNDLoRA)):
            module._mix_trajectories(lora_state_dicts)

    print(model)
    # model.model.set_lora_enabled(False)
    # model.model.set_lora_train_while_rest_is_mode(False)
    results_dict, log_dict = model(features,
                                   loss_fn=nn.CrossEntropyLoss(),
                                   label=torch.LongTensor([1]),
                                   return_attention=True,
                                   return_slide_feats=True
                                   )
    kvs = [(k, v) for k, v in model.state_dict().items()]
    F = 3
