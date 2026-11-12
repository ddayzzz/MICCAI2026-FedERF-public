#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tiantian
"""
import pdb
import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

# typing import
from typing import Dict, Iterable, Optional


class AudioConvRNNEncoder(nn.Module):
    def __init__(self, audio_input_dim, n_filters, d_hid, dropout_p=0.1, reduce=None):

        super(AudioConvRNNEncoder, self).__init__()
        self.dropout_p = dropout_p
        # Conv Encoder module
        self.audio_conv = Conv1dEncoder(
            input_dim=audio_input_dim,
            n_filters=n_filters,
            dropout=self.dropout_p,
        )

        # RNN module
        self.audio_rnn = nn.GRU(
            input_size=n_filters * 4,
            hidden_size=d_hid,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=False
        )
        self.reduce = reduce


    def forward(self, x_audio, len_a, reduce_fill_zero=False):
        # 1. Conv forward
        x_audio = self.audio_conv(x_audio)

        # 2. Rnn forward
        # max pooling, time dim reduce by 8 times
        len_a = len_a // 8
        if reduce_fill_zero:
            len_a[len_a == 0] = 1  # 避免出现==0的情况, 这里
        if len_a[0] != 0:
            try:
                x_audio = pack_padded_sequence(
                    x_audio,
                    len_a.cpu().numpy(),
                    batch_first=True,
                    enforce_sorted=False
                )
            except RuntimeError as e:
                pass

        x_audio, _ = self.audio_rnn(x_audio)
        if len_a[0] != 0:
            x_audio, _ = pad_packed_sequence(
                x_audio,
                batch_first=True
            )
        if self.reduce is None:
            return x_audio
        elif self.reduce == 'mean':
            return torch.mean(x_audio, axis=1)
        else:
            raise NotImplementedError

class VideoFrameRNNEncoder(nn.Module):
    def __init__(self, video_input_dim, d_hid, dropout_p=0.1, reduce=None):

        super(VideoFrameRNNEncoder, self).__init__()
        self.dropout_p = dropout_p
        # Conv Encoder module
        self.video_rnn = nn.GRU(
            input_size=video_input_dim,
            hidden_size=d_hid,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=False
        )
        self.reduce = reduce


    def forward(self, x_video, len_v):
        if len_v[0] != 0:
            x_video = pack_padded_sequence(
                x_video,
                len_v.cpu().numpy(),
                batch_first=True,
                enforce_sorted=False
            )

        x_video, _ = self.video_rnn(x_video)
        if len_v[0] != 0:
            x_video, _ = pad_packed_sequence(
                x_video,
                batch_first=True
            )

        if self.reduce is None:
            return x_video
        elif self.reduce == 'mean':
            return torch.mean(x_video, axis=1)
        else:
            raise NotImplementedError


class MMActionClassifier(nn.Module):
    def __init__(
            self,
            num_classes: int,  # Number of classes
            audio_input_dim: int,  # Audio feature input dim
            video_input_dim: int,  # Frame-wise video feature input dim
            d_hid: int = 128,  # Hidden Layer size
            n_filters: int = 32,  # number of filters
            en_att: bool = False,  # Enable self attention or not
            att_name: str = '',  # Attention Name
            d_head: int = 6,  # Head dim,
            clf_hidden_size=64,
    ):
        super(MMActionClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        self.d_hid = d_hid

        # Conv Encoder module
        self.audio_encoder = AudioConvRNNEncoder(audio_input_dim, n_filters, d_hid, dropout_p=self.dropout_p, reduce=None)
        self.video_encoder = VideoFrameRNNEncoder(video_input_dim, d_hid, dropout_p=self.dropout_p, reduce=None)


        # Attention modules
        if self.att_name == "multihead":
            self.audio_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid,
                num_heads=4,
                dropout=self.dropout_p
            )

            self.video_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid,
                num_heads=4,
                dropout=self.dropout_p
            )
        elif self.att_name == "additive":
            self.audio_att = AdditiveAttention(
                d_hid=d_hid,
                d_att=128
            )
            self.video_att = AdditiveAttention(
                d_hid=d_hid,
                d_att=128
            )
        elif self.att_name == "base":
            self.audio_att = BaseSelfAttention(
                d_hid=d_hid
            )
            self.video_att = BaseSelfAttention(
                d_hid=d_hid
            )
        elif self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(
                d_hid=d_hid,
                d_head=d_head
            )
        elif self.att_name == "hirarchical":
            self.att = HirarchicalAttention(
                d_hid=rnn_input
            )

        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid * d_head, clf_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(clf_hidden_size, num_classes)
            )
        else:
            # Projection head
            # self.audio_proj = nn.Linear(d_hid, d_hid // 2)
            # self.video_proj = nn.Linear(d_hid, d_hid // 2)
            self.classifier = nn.Sequential(
                nn.Linear(d_hid * 2, clf_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(clf_hidden_size, num_classes)
            )

        # Projection head
        # self.init_weight()

    def encode_feature(self, batch_dict,
                       modal_name, reduction='mean'):
        if modal_name == "audio":
            x_audio, x_video, len_a, len_v = batch_dict['audio'], batch_dict['frame'], batch_dict['audio_len'], \
            batch_dict['frame_len']
            x = self.audio_encoder(x_audio, len_a=len_a)
        elif modal_name == "frame":
            x_audio, x_video, len_a, len_v = batch_dict['audio'], batch_dict['frame'], batch_dict['audio_len'], \
                batch_dict['frame_len']
            x = self.video_encoder(x_video, len_v=len_v)
        if reduction == "mean":
            x = x.mean(dim=1)
        return x


    def forward(self, batch_dict, **kwargs):
        x_audio, x_video, len_a, len_v = batch_dict['audio'], batch_dict['frame'], batch_dict['audio_len'], batch_dict['frame_len']
        # encoders output
        # 如果已经存在了必要的特征, 就不会提取!
        x_audio = self.audio_encoder(x_audio, len_a=len_a)
        x_video = self.video_encoder(x_video, len_v=len_v)

        # 3. Attention
        if self.en_att:
            if self.att_name == 'multihead':
                x_audio, _ = self.audio_att(x_audio, x_audio, x_audio)
                x_video, _ = self.video_att(x_video, x_video, x_video)
                # 4. Average pooling
                x_audio = torch.mean(x_audio, axis=1)
                x_video = torch.mean(x_video, axis=1)
            elif self.att_name == 'additive':
                # get attention output
                x_audio = self.audio_att(x_audio, x_audio, x_audio, len_a)
                x_video = self.video_att(x_video, x_video, x_video, len_v)
            elif self.att_name == "fuse_base":
                # get attention output
                a_max_len = x_audio.shape[1]
                x_mm = torch.cat((x_audio, x_video), dim=1)
                x_mm = self.fuse_att(x_mm, len_a, len_v, a_max_len)
            elif self.att_name == 'base':
                # get attention output
                x_audio = self.audio_att(x_audio)
                x_video = self.video_att(x_video, len_v)
        else:
            # 4. Average pooling
            x_audio = torch.mean(x_audio, axis=1)
            x_video = torch.mean(x_video, axis=1)
            # 这里只适合pool的特征, 已经经过均值
            if "audio_pool_feature" in batch_dict:
                assert len(batch_dict['audio_pool_feature']) == len(len_a)
                for fi, filled_feature in enumerate(batch_dict['audio_pool_feature']):
                    if filled_feature is not None:
                        x_audio[fi] = batch_dict['audio_pool_feature'][fi]
            if "frame_pool_feature" in batch_dict:
                assert len(batch_dict['frame_pool_feature']) == len(len_a)
                for fi, filled_feature in enumerate(batch_dict['frame_pool_feature']):
                    if filled_feature is not None:
                        x_video[fi] = batch_dict['frame_pool_feature'][fi]

            x_mm = torch.cat((x_audio, x_video), dim=1)

        # 5. Projection with no attention
        if self.en_att and self.att_name != "fuse_base":
            x_audio = self.audio_proj(x_audio)
            x_video = self.video_proj(x_video)
            x_mm = torch.cat((x_audio, x_video), dim=1)

        # 6. MM embedding and predict
        preds = self.classifier(x_mm)
        return preds

    def load_uni_encodes_state_dict(self, src_model):
        if isinstance(src_model, MMActionSeparationClassifier):
            self.audio_encoder.load_state_dict(src_model.audio_encoder.state_dict())
            self.video_encoder.load_state_dict(src_model.video_encoder.state_dict())
        else:
            raise NotImplementedError(f"Unsupported model: {src_model.__class__.__name__}")

    def forward_head(self, base_feature, **kwargs):
        out = base_feature
        for mod in list(self.classifier.children())[1:]:
            out = mod(out)
        return out
    def forward_base(self, batch_dict, **kwargs):
        x_audio, x_video, len_a, len_v = batch_dict['audio'], batch_dict['frame'], batch_dict['audio_len'], batch_dict['frame_len']
        # encoders output
        x_audio = self.audio_encoder(x_audio, len_a=len_a)
        x_video = self.video_encoder(x_video, len_v=len_v)

        # 3. Attention
        if self.en_att:
            if self.att_name == 'multihead':
                x_audio, _ = self.audio_att(x_audio, x_audio, x_audio)
                x_video, _ = self.video_att(x_video, x_video, x_video)
                # 4. Average pooling
                x_audio = torch.mean(x_audio, axis=1)
                x_video = torch.mean(x_video, axis=1)
            elif self.att_name == 'additive':
                # get attention output
                x_audio = self.audio_att(x_audio, x_audio, x_audio, len_a)
                x_video = self.video_att(x_video, x_video, x_video, len_v)
            elif self.att_name == "fuse_base":
                # get attention output
                a_max_len = x_audio.shape[1]
                x_mm = torch.cat((x_audio, x_video), dim=1)
                x_mm = self.fuse_att(x_mm, len_a, len_v, a_max_len)
            elif self.att_name == 'base':
                # get attention output
                x_audio = self.audio_att(x_audio)
                x_video = self.video_att(x_video, len_v)
        else:
            # 4. Average pooling
            x_audio = torch.mean(x_audio, axis=1)
            x_video = torch.mean(x_video, axis=1)
            x_mm = torch.cat((x_audio, x_video), dim=1)

        # 5. Projection with no attention
        if self.en_att and self.att_name != "fuse_base":
            x_audio = self.audio_proj(x_audio)
            x_video = self.video_proj(x_video)
            x_mm = torch.cat((x_audio, x_video), dim=1)
        # 6. MM embedding and predict
        preds = self.classifier[0](x_mm)
        return preds

    def forward_classifier(self, mm_features):
        return self.classifier(mm_features)

class MMActionSeparationClassifier(nn.Module):
    def __init__(
            self,
            num_classes: int,  # Number of classes
            audio_input_dim: int,  # Audio feature input dim
            video_input_dim: int,  # Frame-wise video feature input dim
            d_hid: int = 128,  # Hidden Layer size
            n_filters: int = 32,  # number of filters
            en_att: bool = False,  # Enable self attention or not
            att_name: str = '',  # Attention Name
            d_head: int = 6,  # Head dim,
            clf_hidden_size=64,
    ):
        super(MMActionSeparationClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name

        # Conv Encoder module
        self.audio_encoder = AudioConvRNNEncoder(audio_input_dim, n_filters, d_hid, dropout_p=self.dropout_p, reduce=None)
        self.video_encoder = VideoFrameRNNEncoder(video_input_dim, d_hid, dropout_p=self.dropout_p, reduce=None)


        # Attention modules
        if self.att_name == "multihead":
            self.audio_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid,
                num_heads=4,
                dropout=self.dropout_p
            )

            self.video_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid,
                num_heads=4,
                dropout=self.dropout_p
            )
        elif self.att_name == "additive":
            self.audio_att = AdditiveAttention(
                d_hid=d_hid,
                d_att=128
            )
            self.video_att = AdditiveAttention(
                d_hid=d_hid,
                d_att=128
            )
        elif self.att_name == "base":
            self.audio_att = BaseSelfAttention(
                d_hid=d_hid
            )
            self.video_att = BaseSelfAttention(
                d_hid=d_hid
            )
        elif self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(
                d_hid=d_hid,
                d_head=d_head
            )
        elif self.att_name == "hirarchical":
            self.att = HirarchicalAttention(
                d_hid=rnn_input
            )

        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid * d_head, clf_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(clf_hidden_size, num_classes)
            )
        else:
            # Projection head
            # self.audio_proj = nn.Linear(d_hid, d_hid // 2)
            # self.video_proj = nn.Linear(d_hid, d_hid // 2)
            self.video_classifier = nn.Sequential(
                nn.Linear(d_hid, clf_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(clf_hidden_size, num_classes)
            )
            self.audio_classifier = nn.Sequential(
                nn.Linear(d_hid, clf_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(clf_hidden_size, num_classes)
            )

        # Projection head
        # self.init_weight()

    # def init_weight(self):
    #     for m in self._modules:
    #         if type(m) == nn.Linear:
    #             torch.nn.init.xavier_uniform(m.weight)
    #             m.bias.data.fill_(0.01)
    #         if type(m) == nn.Conv1d:
    #             torch.nn.init.xavier_uniform(m.weight)
    #             m.bias.data.fill_(0.01)

    # def forward(
    #         self,
    #         x_audio,
    #         x_video,
    #         len_a,
    #         len_v
    # ):
    def forward(self, batch_dict, **kwargs):
        x_audio, x_video, len_a, len_v = batch_dict['audio'], batch_dict['frame'], batch_dict['audio_len'], batch_dict['frame_len']
        # encoders output
        x_audio = self.audio_encoder(x_audio, len_a=len_a)
        x_video = self.video_encoder(x_video, len_v=len_v)

        # # 3. Attention
        # if self.en_att:
        #     if self.att_name == 'multihead':
        #         x_audio, _ = self.audio_att(x_audio, x_audio, x_audio)
        #         x_video, _ = self.video_att(x_video, x_video, x_video)
        #         # 4. Average pooling
        #         x_audio = torch.mean(x_audio, axis=1)
        #         x_video = torch.mean(x_video, axis=1)
        #     elif self.att_name == 'additive':
        #         # get attention output
        #         x_audio = self.audio_att(x_audio, x_audio, x_audio, len_a)
        #         x_video = self.video_att(x_video, x_video, x_video, len_v)
        #     elif self.att_name == "fuse_base":
        #         # get attention output
        #         a_max_len = x_audio.shape[1]
        #         x_mm = torch.cat((x_audio, x_video), dim=1)
        #         x_mm = self.fuse_att(x_mm, len_a, len_v, a_max_len)
        #     elif self.att_name == 'base':
        #         # get attention output
        #         x_audio = self.audio_att(x_audio)
        #         x_video = self.video_att(x_video, len_v)
        # else:
        #     # 4. Average pooling
        #     x_audio = torch.mean(x_audio, axis=1)
        #     x_video = torch.mean(x_video, axis=1)
        #     x_mm = torch.cat((x_audio, x_video), dim=1)
        #
        # # 5. Projection with no attention
        # if self.en_att and self.att_name != "fuse_base":
        #     x_audio = self.audio_proj(x_audio)
        #     x_video = self.video_proj(x_video)
        #     x_mm = torch.cat((x_audio, x_video), dim=1)
        # # 6. MM embedding and predict
        # preds = self.classifier(x_mm)
        # return preds
        x_audio = torch.mean(x_audio, axis=1)
        x_video = torch.mean(x_video, axis=1)
        audio_preds = self.audio_classifier(x_audio)
        video_preds = self.video_classifier(x_video)
        return audio_preds, video_preds

    def forward_head(self, base_feature, **kwargs):
        out = base_feature
        for mod in list(self.classifier.children())[1:]:
            out = mod(out)
        return out

    def forward_base(self, batch_dict, **kwargs):
        x_audio, x_video, len_a, len_v = batch_dict['audio'], batch_dict['frame'], batch_dict['audio_len'], batch_dict['frame_len']
        # encoders output
        x_audio = self.audio_encoder(x_audio, len_a=len_a)
        x_video = self.video_encoder(x_video, len_v=len_v)

        # 3. Attention
        if self.en_att:
            if self.att_name == 'multihead':
                x_audio, _ = self.audio_att(x_audio, x_audio, x_audio)
                x_video, _ = self.video_att(x_video, x_video, x_video)
                # 4. Average pooling
                x_audio = torch.mean(x_audio, axis=1)
                x_video = torch.mean(x_video, axis=1)
            elif self.att_name == 'additive':
                # get attention output
                x_audio = self.audio_att(x_audio, x_audio, x_audio, len_a)
                x_video = self.video_att(x_video, x_video, x_video, len_v)
            elif self.att_name == "fuse_base":
                # get attention output
                a_max_len = x_audio.shape[1]
                x_mm = torch.cat((x_audio, x_video), dim=1)
                x_mm = self.fuse_att(x_mm, len_a, len_v, a_max_len)
            elif self.att_name == 'base':
                # get attention output
                x_audio = self.audio_att(x_audio)
                x_video = self.video_att(x_video, len_v)
        else:
            # 4. Average pooling
            x_audio = torch.mean(x_audio, axis=1)
            x_video = torch.mean(x_video, axis=1)
            x_mm = torch.cat((x_audio, x_video), dim=1)

        # 5. Projection with no attention
        if self.en_att and self.att_name != "fuse_base":
            x_audio = self.audio_proj(x_audio)
            x_video = self.video_proj(x_video)
            x_mm = torch.cat((x_audio, x_video), dim=1)
        # 6. MM embedding and predict
        preds = self.classifier[0](x_mm)
        return preds


class SERClassifier(nn.Module):
    def __init__(
            self,
            num_classes: int,  # Number of classes
            audio_input_dim: int,  # Audio data input dim
            text_input_dim: int,  # Text data input dim
            d_hid: int = 64,  # Hidden Layer size
            n_filters: int = 32,  # number of filters
            en_att: bool = False,  # Enable self attention or not
            att_name: str = '',  # Attention Name
            d_head: int = 6,  # Head dim
            clf_hidden_size = 64,
    ):
        super(SERClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        self.d_hid = d_hid

        # Conv Encoder module
        self.audio_encoder = AudioConvRNNEncoder(audio_input_dim, n_filters, d_hid, dropout_p=self.dropout_p, reduce=None)
        self.text_encoder = UniTextGRUEncoder(text_input_dim, d_hid=d_hid, dropout_p=self.dropout_p, hidden_reduction=None)

        # Self attention module
        if self.att_name == "multihead":
            self.audio_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid,
                num_heads=4,
                dropout=self.dropout_p
            )
            self.text_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid,
                num_heads=4,
                dropout=self.dropout_p
            )
        elif self.att_name == "base":
            self.audio_att = BaseSelfAttention(
                d_hid=d_hid
            )
            self.text_att = BaseSelfAttention(
                d_hid=d_hid
            )
        elif self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(
                d_hid=d_hid,
                d_head=d_head
            )

        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid * d_head, clf_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(clf_hidden_size, num_classes)
            )
        else:
            # Projection head
            # self.audio_proj = nn.Linear(d_hid, d_hid // 2)
            # self.text_proj = nn.Linear(d_hid, d_hid // 2)
            # self.init_weight()

            # classifier head
            self.classifier = nn.Sequential(
                nn.Linear(d_hid * 2, clf_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(clf_hidden_size, num_classes)
            )

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def encode_feature(self, batch_dict,
                       modal_name, reduction='mean'):
        if modal_name == "audio":
            x_audio, len_a = batch_dict['audio'], batch_dict['audio_len']
            x = self.audio_encoder(x_audio, len_a=len_a, reduce_fill_zero=True)
            x = x.mean(dim=1)
        elif modal_name == "text":
            x_text, len_t = batch_dict['text'], batch_dict['text_len']
            x = self.text_encoder(x_text, len_t=len_t)
            x = x.mean(dim=1)
        return x

    def forward(self, batch_dict, **kwargs):
        x_audio, x_text, len_a, len_t =  batch_dict['audio'], batch_dict['text'], batch_dict['audio_len'], batch_dict['text_len']
        x_audio = self.audio_encoder(x_audio, len_a=len_a, reduce_fill_zero=True)
        x_text = self.text_encoder(x_text, len_t=len_t)
        # 3. Attention
        if self.en_att:
            if self.att_name == 'multihead':
                x_audio, _ = self.audio_att(x_audio, x_audio, x_audio)
                x_text, _ = self.text_att(x_text, x_text, x_text)
                # 4. Average pooling
                x_audio = torch.mean(x_audio, axis=1)
                x_text = torch.mean(x_text, axis=1)
            elif self.att_name == 'base':
                # get attention output
                x_audio = self.audio_att(x_audio)
                x_text = self.text_att(x_text, l_b)
            elif self.att_name == "fuse_base":
                # get attention output
                a_max_len = x_audio.shape[1]
                x_mm = torch.cat((x_audio, x_text), dim=1)
                x_mm = self.fuse_att(x_mm, len_a, len_t, a_max_len)
        else:
            # 4. Average pooling Projection
            x_audio = torch.mean(x_audio, axis=1)
            x_text = torch.mean(x_text, axis=1)
            # 这里只适合pool的特征, 已经经过均值
            if "audio_pool_feature" in batch_dict:
                assert len(batch_dict['audio_pool_feature']) == len(len_a)
                for fi, filled_feature in enumerate(batch_dict['audio_pool_feature']):
                    if filled_feature is not None:
                        x_audio[fi] = batch_dict['audio_pool_feature'][fi]
            if "text_pool_feature" in batch_dict:
                assert len(batch_dict['text_pool_feature']) == len(len_t)
                for fi, filled_feature in enumerate(batch_dict['text_pool_feature']):
                    if filled_feature is not None:
                        x_text[fi] = batch_dict['text_pool_feature'][fi]

            x_mm = torch.cat((x_audio, x_text), dim=1)

        # 5. Projection
        if self.en_att and self.att_name != "fuse_base":
            x_audio = self.audio_proj(x_audio)
            x_text = self.text_proj(x_text)
            x_mm = torch.cat((x_audio, x_text), dim=1)



        # 6. MM embedding and predict
        preds = self.classifier(x_mm)
        return preds # , x_mm

    def forward_head(self, base_feature, **kwargs):
        out = base_feature
        for mod in list(self.classifier.children())[1:]:
            out = mod(out)
        return out

    def forward_base(self, batch_dict, **kwargs):
        x_audio, x_text, len_a, len_t =  batch_dict['audio'], batch_dict['text'], batch_dict['audio_len'], batch_dict['text_len']
        x_audio = self.audio_encoder(x_audio, len_a=len_a, reduce_fill_zero=True)
        x_text = self.text_encoder(x_text, len_t=len_t)
        # 3. Attention
        if self.en_att:
            if self.att_name == 'multihead':
                x_audio, _ = self.audio_att(x_audio, x_audio, x_audio)
                x_text, _ = self.text_att(x_text, x_text, x_text)
                # 4. Average pooling
                x_audio = torch.mean(x_audio, axis=1)
                x_text = torch.mean(x_text, axis=1)
            elif self.att_name == 'base':
                # get attention output
                x_audio = self.audio_att(x_audio)
                x_text = self.text_att(x_text, l_b)
            elif self.att_name == "fuse_base":
                # get attention output
                a_max_len = x_audio.shape[1]
                x_mm = torch.cat((x_audio, x_text), dim=1)
                x_mm = self.fuse_att(x_mm, len_a, len_t, a_max_len)
        else:
            # 4. Average pooling Projection
            x_audio = torch.mean(x_audio, axis=1)
            x_text = torch.mean(x_text, axis=1)
            x_mm = torch.cat((x_audio, x_text), dim=1)

        # 5. Projection
        if self.en_att and self.att_name != "fuse_base":
            x_audio = self.audio_proj(x_audio)
            x_text = self.text_proj(x_text)
            x_mm = torch.cat((x_audio, x_text), dim=1)

        # 6. MM embedding and predict
        latent = self.classifier[0](x_mm)
        return latent # , x_mm
    def forward_classifier(self, mm_features):
        return self.classifier(mm_features)
    def load_uni_encodes_state_dict(self, src_model):
        if isinstance(src_model, SERSeparateClassifiers):
            self.audio_encoder.load_state_dict(src_model.audio_encoder.state_dict())
            self.text_encoder.load_state_dict(src_model.text_encoder.state_dict())
        else:
            raise NotImplementedError(f"Unsupported model: {src_model.__class__.__name__}")

class SERSeparateClassifiers(nn.Module):
    def __init__(
            self,
            num_classes: int,  # Number of classes
            audio_input_dim: int,  # Audio data input dim
            text_input_dim: int,  # Text data input dim
            d_hid: int = 64,  # Hidden Layer size
            n_filters: int = 32,  # number of filters
            en_att: bool = False,  # Enable self attention or not
            att_name: str = '',  # Attention Name
            d_head: int = 6,  # Head dim
            clf_hidden_size = 64,
    ):
        super(SERSeparateClassifiers, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name

        # Conv Encoder module
        self.audio_encoder = AudioConvRNNEncoder(audio_input_dim, n_filters, d_hid, dropout_p=self.dropout_p, reduce=None)
        self.text_encoder = UniTextGRUEncoder(text_input_dim, d_hid=d_hid, dropout_p=self.dropout_p, hidden_reduction=None)

        # Self attention module
        if self.att_name == "multihead":
            self.audio_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid,
                num_heads=4,
                dropout=self.dropout_p
            )
            self.text_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid,
                num_heads=4,
                dropout=self.dropout_p
            )
        elif self.att_name == "base":
            self.audio_att = BaseSelfAttention(
                d_hid=d_hid
            )
            self.text_att = BaseSelfAttention(
                d_hid=d_hid
            )
        elif self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(
                d_hid=d_hid,
                d_head=d_head
            )

        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid * d_head, clf_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(clf_hidden_size, num_classes)
            )
        else:
            # Projection head
            self.audio_proj = nn.Linear(d_hid, d_hid // 2)
            self.text_proj = nn.Linear(d_hid, d_hid // 2)
            # self.init_weight()

            # classifier head
            self.audio_classifier = nn.Sequential(
                nn.Linear(d_hid, clf_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(clf_hidden_size, num_classes)
            )
            self.text_classifier = nn.Sequential(
                nn.Linear(d_hid, clf_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(clf_hidden_size, num_classes)
            )

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, batch_dict, **kwargs):
        x_audio, x_text, len_a, len_t =  batch_dict['audio'], batch_dict['text'], batch_dict['audio_len'], batch_dict['text_len']
        x_audio = self.audio_encoder(x_audio, len_a=len_a, reduce_fill_zero=True)
        x_text = self.text_encoder(x_text, len_t=len_t)
        # 3. Attention
        assert not self.en_att
        # if self.en_att:
        #     if self.att_name == 'multihead':
        #         x_audio, _ = self.audio_att(x_audio, x_audio, x_audio)
        #         x_text, _ = self.text_att(x_text, x_text, x_text)
        #         # 4. Average pooling
        #         x_audio = torch.mean(x_audio, axis=1)
        #         x_text = torch.mean(x_text, axis=1)
        #     elif self.att_name == 'base':
        #         # get attention output
        #         x_audio = self.audio_att(x_audio)
        #         x_text = self.text_att(x_text, l_b)
        #     elif self.att_name == "fuse_base":
        #         # get attention output
        #         a_max_len = x_audio.shape[1]
        #         x_mm = torch.cat((x_audio, x_text), dim=1)
        #         x_mm = self.fuse_att(x_mm, len_a, len_t, a_max_len)
        # else:
        #     # 4. Average pooling Projection
        #     x_audio = torch.mean(x_audio, axis=1)
        #     x_text = torch.mean(x_text, axis=1)
        #     x_mm = torch.cat((x_audio, x_text), dim=1)

        # 5. Projection
        # if self.en_att and self.att_name != "fuse_base":
        #     x_audio = self.audio_proj(x_audio)
        #     x_text = self.text_proj(x_text)
        #     x_mm = torch.cat((x_audio, x_text), dim=1)

        # 6. MM embedding and predict
        # preds = self.classifier(x_mm)
        # return preds # , x_mm
        x_audio = torch.mean(x_audio, axis=1)
        x_text = torch.mean(x_text, axis=1)
        a_preds = self.audio_classifier(x_audio)
        t_preds = self.text_classifier(x_text)
        return a_preds, t_preds

    def forward_head(self, base_feature, **kwargs):
        out = base_feature
        for mod in list(self.classifier.children())[1:]:
            out = mod(out)
        return out

    def forward_base(self, batch_dict, **kwargs):
        x_audio, x_text, len_a, len_t =  batch_dict['audio'], batch_dict['text'], batch_dict['audio_len'], batch_dict['text_len']
        x_audio = self.audio_encoder(x_audio, len_a=len_a, reduce_fill_zero=True)
        x_text = self.text_encoder(x_text, len_t=len_t)
        # 3. Attention
        if self.en_att:
            if self.att_name == 'multihead':
                x_audio, _ = self.audio_att(x_audio, x_audio, x_audio)
                x_text, _ = self.text_att(x_text, x_text, x_text)
                # 4. Average pooling
                x_audio = torch.mean(x_audio, axis=1)
                x_text = torch.mean(x_text, axis=1)
            elif self.att_name == 'base':
                # get attention output
                x_audio = self.audio_att(x_audio)
                x_text = self.text_att(x_text, l_b)
            elif self.att_name == "fuse_base":
                # get attention output
                a_max_len = x_audio.shape[1]
                x_mm = torch.cat((x_audio, x_text), dim=1)
                x_mm = self.fuse_att(x_mm, len_a, len_t, a_max_len)
        else:
            # 4. Average pooling Projection
            x_audio = torch.mean(x_audio, axis=1)
            x_text = torch.mean(x_text, axis=1)
            x_mm = torch.cat((x_audio, x_text), dim=1)

        # 5. Projection
        if self.en_att and self.att_name != "fuse_base":
            x_audio = self.audio_proj(x_audio)
            x_text = self.text_proj(x_text)
            x_mm = torch.cat((x_audio, x_text), dim=1)

        # 6. MM embedding and predict
        latent = self.classifier[0](x_mm)
        return latent # , x_mm


class UniTextGRUEncoder(nn.Module):
    def __init__(self, text_input_dim, d_hid, dropout_p, hidden_reduction=None):
        super(UniTextGRUEncoder, self).__init__()
        self.text_rnn = nn.GRU(
            input_size=text_input_dim,
            hidden_size=d_hid,
            num_layers=1,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False
        )
        self.hidden_reduction = hidden_reduction

    def forward(self, x_text, len_t):
        if len_t[0] != 0:
            x_text = pack_padded_sequence(
                x_text,
                len_t.cpu().numpy(),
                batch_first=True,
                enforce_sorted=False
            )
        x_text, _ = self.text_rnn(x_text)
        if len_t[0] != 0:
            x_text, _ = pad_packed_sequence(x_text, batch_first=True)
        if self.hidden_reduction is not None:
            if self.hidden_reduction == "mean":
                return torch.mean(x_text, axis=1)
            else:
                raise NotImplementedError("Unsupported hidden reduction: %s" % self.hidden_reduction)
        return x_text

class ImageTextClassifier(nn.Module):
    def __init__(
            self,
            num_classes: int,  # Number of classes
            img_input_dim: int,  # Image data input dim
            text_input_dim: int,  # Text data input dim
            d_hid: int = 64,  # Hidden Layer size
            en_att: bool = False,  # Enable self attention or not
            att_name: str = '',  # Attention Name
            d_head: int = 6,  # Head dim
            clf_hidden_size=64,
    ):
        super(ImageTextClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        self.d_hid = d_hid

        # Projection head
        self.img_proj = nn.Sequential(
            nn.Linear(img_input_dim, d_hid),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(d_hid, d_hid)
        )

        # RNN module
        self.text_rnn = UniTextGRUEncoder(text_input_dim, d_hid, dropout_p=self.dropout_p)

        # Self attention module
        if self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(
                d_hid=d_hid,
                d_head=d_head
            )

        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid * d_head, clf_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(clf_hidden_size, num_classes)
            )
        else:
            # classifier head
            self.classifier = nn.Sequential(
                nn.Linear(d_hid * 2, clf_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(clf_hidden_size, num_classes)
            )

        # self.init_weight()

    def forward_classifier(self, mm_features):
        return self.classifier(mm_features)

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def encode_feature(self, batch_dict,
                       modal_name,):
        if modal_name == "image":
            x_img, len_i = batch_dict['image'], batch_dict['image_len']
            x = self.img_proj(x_img[:, 0, :])
        elif modal_name == "text":
            x_text, len_t = batch_dict['text'], batch_dict['text_len']
            x = self.text_rnn(x_text, len_t)
            x = x.mean(dim=1)

        return x

    def forward(self, batch_dict, **kwargs):
        x_img, x_text, len_i, len_t = batch_dict['image'], batch_dict['text'], batch_dict['image_len'], batch_dict['text_len']
        # 1. img proj
        x_img = self.img_proj(x_img[:, 0, :])

        # 2. Rnn forward
        # print(f"len_t: {len_t}")
        x_text = self.text_rnn(x_text, len_t)

        # 3. Attention
        if self.en_att:
            if self.att_name == "fuse_base":
                # get attention output
                x_mm = torch.cat((x_img.unsqueeze(dim=1), x_text), dim=1)
                x_mm = self.fuse_att(x_mm, len_i, len_t, 1)
        else:
            # 4. Average pooling
            x_text = torch.mean(x_text, axis=1)

            # 这里只适合pool的特征, 已经经过均值
            if "image_pool_feature" in batch_dict:
                assert len(batch_dict['image_pool_feature']) == len(len_t)
                for fi, filled_feature in enumerate(batch_dict['image_pool_feature']):
                    if filled_feature is not None:
                        x_img[fi] = batch_dict['image_pool_feature'][fi]
            if "text_pool_feature" in batch_dict:
                assert len(batch_dict['text_pool_feature']) == len(len_t)
                for fi, filled_feature in enumerate(batch_dict['text_pool_feature']):
                    if filled_feature is not None:
                        x_text[fi] = batch_dict['text_pool_feature'][fi]

            x_mm = torch.cat((x_img, x_text), dim=1)

        # 4. MM embedding and predict
        preds = self.classifier(x_mm)
        return preds

    def forward_base(self, batch_dict, **kwargs):
        x_img, x_text, len_i, len_t = batch_dict['image'], batch_dict['text'], batch_dict['image_len'], batch_dict[
            'text_len']
        # 1. img proj
        x_img = self.img_proj(x_img[:, 0, :])

        # 2. Rnn forward
        # print(f"len_t: {len_t}")
        x_text = self.text_rnn(x_text, len_t)

        # 3. Attention
        if self.en_att:
            if self.att_name == "fuse_base":
                # get attention output
                x_mm = torch.cat((x_img.unsqueeze(dim=1), x_text), dim=1)
                x_mm = self.fuse_att(x_mm, len_i, len_t, 1)
        else:
            # 4. Average pooling
            x_text = torch.mean(x_text, axis=1)
            x_mm = torch.cat((x_img, x_text), dim=1)

        # 4. MM embedding and predict
        preds = self.classifier[0](x_mm)
        return preds

    def forward_head(self, base_feature, **kwargs):
        out = base_feature
        for mod in list(self.classifier.children())[1:]:
            out = mod(out)
        return out

    def load_uni_encodes_state_dict(self, src_model):
        if isinstance(src_model, ImageTextSeparateClassifier):
            self.img_proj.load_state_dict(src_model.img_proj.state_dict())
            self.text_rnn.load_state_dict(src_model.text_rnn.state_dict())
        else:
            raise NotImplementedError(f"Unsupported model: {src_model.__class__.__name__}")

class ImageTextSeparateClassifier(nn.Module):
    def __init__(
            self,
            num_classes: int,  # Number of classes
            img_input_dim: int,  # Image data input dim
            text_input_dim: int,  # Text data input dim
            d_hid: int = 64,  # Hidden Layer size
            en_att: bool = False,  # Enable self attention or not
            att_name: str = '',  # Attention Name
            d_head: int = 6,  # Head dim
            clf_hidden_size=64,
    ):
        super(ImageTextSeparateClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name

        # Projection head
        self.img_proj = nn.Sequential(
            nn.Linear(img_input_dim, d_hid),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(d_hid, d_hid)
        )

        # RNN module
        self.text_rnn = UniTextGRUEncoder(text_input_dim, d_hid, dropout_p=self.dropout_p)

        # Self attention module
        if self.att_name == "fuse_base":
            # self.fuse_att = FuseBaseSelfAttention(
            #     d_hid=d_hid,
            #     d_head=d_head
            # )
            raise NotImplementedError()

        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            # self.classifier = nn.Sequential(
            #     nn.Linear(d_hid * d_head, clf_hidden_size),
            #     nn.ReLU(),
            #     nn.Dropout(self.dropout_p),
            #     nn.Linear(clf_hidden_size, num_classes)
            # )
            raise NotImplementedError()
        else:
            # classifier head
            self.image_classifier = nn.Sequential(
                nn.Linear(d_hid, clf_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(clf_hidden_size, num_classes)
            )
            self.text_classifier = nn.Sequential(
                nn.Linear(d_hid, clf_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(clf_hidden_size, num_classes)
            )

        # self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, batch_dict, **kwargs):
        x_img, x_text, len_i, len_t = batch_dict['image'], batch_dict['text'], batch_dict['image_len'], batch_dict['text_len']
        # 1. img proj
        x_img = self.img_proj(x_img[:, 0, :])

        # 2. Rnn forward
        # print(f"len_t: {len_t}")
        x_text = self.text_rnn(x_text, len_t)

        # # 3. Attention
        # if self.en_att:
        #     if self.att_name == "fuse_base":
        #         # get attention output
        #         x_mm = torch.cat((x_img.unsqueeze(dim=1), x_text), dim=1)
        #         x_mm = self.fuse_att(x_mm, len_i, len_t, 1)
        # else:
        #     # 4. Average pooling
        #     x_text = torch.mean(x_text, axis=1)
        #     x_mm = torch.cat((x_img, x_text), dim=1)
        #
        # # 4. MM embedding and predict
        # preds = self.classifier(x_mm)
        # return preds

        x_text = torch.mean(x_text, axis=1)
        text_preds = self.text_classifier(x_text)
        image_preds = self.image_classifier(x_img)
        return image_preds, text_preds

    def forward_base(self, batch_dict, **kwargs):
        x_img, x_text, len_i, len_t = batch_dict['image'], batch_dict['text'], batch_dict['image_len'], batch_dict[
            'text_len']
        # 1. img proj
        x_img = self.img_proj(x_img[:, 0, :])

        # 2. Rnn forward
        # print(f"len_t: {len_t}")
        x_text = self.text_rnn(x_text, len_t)

        # 3. Attention
        if self.en_att:
            if self.att_name == "fuse_base":
                # get attention output
                x_mm = torch.cat((x_img.unsqueeze(dim=1), x_text), dim=1)
                x_mm = self.fuse_att(x_mm, len_i, len_t, 1)
        else:
            # 4. Average pooling
            x_text = torch.mean(x_text, axis=1)
            x_mm = torch.cat((x_img, x_text), dim=1)

        # 4. MM embedding and predict
        preds = self.classifier[0](x_mm)
        return preds

    def forward_head(self, base_feature, **kwargs):
        out = base_feature
        for mod in list(self.classifier.children())[1:]:
            out = mod(out)
        return out


class HARClassifier(nn.Module):
    def __init__(
            self,
            num_classes: int,  # Number of classes
            acc_input_dim: int,  # Acc data input dim
            gyro_input_dim: int,  # Gyro data input dim
            d_hid: int = 128,  # Hidden Layer size
            n_filters: int = 32,  # number of filters
            en_att: bool = False,  # Enable self attention or not
            att_name: str = '',  # Attention Name
            d_head: int = 6  # Head dim
    ):
        super(HARClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name

        # Conv Encoder module
        self.acc_conv = Conv1dEncoder(
            input_dim=acc_input_dim,
            n_filters=n_filters,
            dropout=self.dropout_p,
        )

        self.gyro_conv = Conv1dEncoder(
            input_dim=acc_input_dim,
            n_filters=n_filters,
            dropout=self.dropout_p,
        )

        # RNN module
        self.acc_rnn = nn.GRU(
            input_size=n_filters * 4,
            hidden_size=d_hid,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=False
        )

        self.gyro_rnn = nn.GRU(
            input_size=n_filters * 4,
            hidden_size=d_hid,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=False
        )

        # Self attention module
        if self.att_name == "multihead":
            self.acc_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid,
                num_heads=4,
                dropout=self.dropout_p
            )

            self.gyro_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid,
                num_heads=4,
                dropout=self.dropout_p
            )
        elif self.att_name == "base":
            self.acc_att = BaseSelfAttention(d_hid=d_hid)
            self.gyro_att = BaseSelfAttention(d_hid=d_hid)
        elif self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(
                d_hid=d_hid,
                d_head=d_head
            )

        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid * d_head, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        else:
            # Projection head
            self.acc_proj = nn.Linear(d_hid, d_hid // 2)
            self.gyro_proj = nn.Linear(d_hid, d_hid // 2)

            # Classifier head
            self.classifier = nn.Sequential(
                nn.Linear(d_hid * 2, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )

        # self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_acc, x_gyro, l_a, l_b):
        # 1. Conv forward
        x_acc = self.acc_conv(x_acc)
        x_gyro = self.gyro_conv(x_gyro)
        # 2. Rnn forward
        x_acc, _ = self.acc_rnn(x_acc)
        x_gyro, _ = self.gyro_rnn(x_gyro)

        # Length of the signal
        l_a = l_a // 8
        l_b = l_b // 8

        # 3. Attention
        if self.en_att:
            if self.att_name == 'multihead':
                x_acc, _ = self.acc_att(x_acc, x_acc, x_acc)
                x_gyro, _ = self.gyro_att(x_gyro, x_gyro, x_gyro)
                # 4. Average pooling
                x_acc = torch.mean(x_acc, axis=1)
                x_gyro = torch.mean(x_gyro, axis=1)
            elif self.att_name == 'base':
                # get attention output
                x_acc = self.acc_att(x_acc)
                x_gyro = self.gyro_att(x_gyro)
            elif self.att_name == "fuse_base":
                # get attention output
                x_mm = torch.cat((x_acc, x_gyro), dim=1)
                x_mm = self.fuse_att(
                    x_mm,
                    val_a=l_a,
                    val_b=l_b,
                    a_len=x_acc.shape[1]
                )
        else:
            # 4. Average pooling
            x_acc = torch.mean(x_acc, axis=1)
            x_gyro = torch.mean(x_gyro, axis=1)
            x_mm = torch.cat((x_acc, x_gyro), dim=1)

        # 5. Projection
        if self.en_att and self.att_name != "fuse_base":
            x_acc = self.acc_proj(x_acc)
            x_gyro = self.gyro_proj(x_gyro)
            x_mm = torch.cat((x_acc, x_gyro), dim=1)

        # 6. MM embedding and predict
        preds = self.classifier(x_mm)
        return preds, x_mm


class ECGClassifier(nn.Module):
    def __init__(
            self,
            num_classes: int,  # Number of classes
            i_to_avf_input_dim: int,  # 6 lead ecg
            v1_to_v6_input_dim: int,  # v1-v6 ecg
            d_hid: int = 64,  # Hidden Layer size
            n_filters: int = 32,  # number of filters
            en_att: bool = False,  # Enable self attention or not
            att_name: str = '',  # Attention Name
            d_head: int = 6  # Head dim
    ):
        super(ECGClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name

        # Conv Encoder module
        self.i_to_avf_conv = Conv1dEncoder(
            input_dim=i_to_avf_input_dim,
            n_filters=n_filters,
            dropout=self.dropout_p,
        )

        self.v1_to_v6_conv = Conv1dEncoder(
            input_dim=v1_to_v6_input_dim,
            n_filters=n_filters,
            dropout=self.dropout_p,
        )

        # RNN module
        self.i_to_avf_rnn = nn.GRU(
            input_size=n_filters * 4,
            hidden_size=d_hid,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=False
        )

        self.v1_to_v6_rnn = nn.GRU(
            input_size=n_filters * 4,
            hidden_size=d_hid,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=False
        )

        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(
                d_hid=d_hid,
                d_head=d_head
            )
            self.classifier = nn.Sequential(
                nn.Linear(d_hid * d_head, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        else:
            # Projection head
            self.i_to_avf_proj = nn.Linear(d_hid, d_hid // 2)
            self.v1_to_v6_proj = nn.Linear(d_hid, d_hid // 2)

            # Classifier head
            self.classifier = nn.Sequential(
                nn.Linear(d_hid * 2, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )

        # self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_i_to_avf, x_v1_to_v6, l_a, l_b):
        # 1. Conv forward
        x_i_to_avf = self.i_to_avf_conv(x_i_to_avf)
        x_v1_to_v6 = self.v1_to_v6_conv(x_v1_to_v6)

        l_a = l_a // 8
        l_b = l_b // 8

        # 2. Rnn forward
        x_i_to_avf, _ = self.i_to_avf_rnn(x_i_to_avf)
        x_v1_to_v6, _ = self.v1_to_v6_rnn(x_v1_to_v6)
        # 3. Attention
        if self.en_att:
            # get attention output
            x_mm = torch.cat((x_i_to_avf, x_v1_to_v6), dim=1)
            x_mm = self.fuse_att(
                x_mm,
                val_a=l_a,
                val_b=l_b,
                a_len=x_i_to_avf.shape[1]
            )
        else:
            # 4. Average pooling
            x_i_to_avf = torch.mean(x_i_to_avf, axis=1)
            x_v1_to_v6 = torch.mean(x_v1_to_v6, axis=1)
            # 6. MM embedding and predict
            x_mm = torch.cat((x_i_to_avf, x_v1_to_v6), dim=1)
        preds = self.classifier(x_mm)
        return preds, x_mm


class Conv1dEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            n_filters: int,
            dropout: float = 0.1
    ):
        super().__init__()
        # conv module
        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(n_filters * 2, n_filters * 4, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: Tensor  # shape => [batch_size (B), num_data (T), feature_dim (D)]
    ):
        x = x.float()
        x = x.permute(0, 2, 1)
        # conv1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        # conv2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        # conv3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x


def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(
            self,
            d_hid: int = 64,
            d_att: int = 256
    ):
        super().__init__()

        self.query_proj = nn.Linear(d_hid, d_att, bias=False)
        self.key_proj = nn.Linear(d_hid, d_att, bias=False)
        self.bias = nn.Parameter(torch.rand(d_att).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(d_att, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            valid_lens: Tensor
    ):
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        # attn = F.softmax(score, dim=-1)
        attn = masked_softmax(scores, valid_lens)
        attn = self.dropout(attn)
        output = torch.bmm(attn.unsqueeze(1), value)
        return output


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """

    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            pdb.set_trace()
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class HirarchicalAttention(nn.Module):
    '''
    ref: Hierarchical Attention Networks
    '''

    def __init__(self, d_hid: int):
        super(HirarchicalAttention, self).__init__()
        self.w_linear = nn.Linear(d_hid, d_hid)
        self.u_w = nn.Linear(d_hid, 1, bias=False)

    def forward(self, input: torch.Tensor):
        u_it = torch.tanh(self.w_linear(input))
        a_it = torch.softmax(self.u_w(u_it), dim=1)
        s_i = input * a_it
        return s_i


class HirarchicalAttention(nn.Module):
    '''
    ref: Hierarchical Attention Networks
    '''

    def __init__(self, d_hid: int):
        super(HirarchicalAttention, self).__init__()
        self.w_linear = nn.Linear(d_hid, d_hid)
        self.u_w = nn.Linear(d_hid, 1, bias=False)

    def forward(self, input: torch.Tensor):
        u_it = torch.tanh(self.w_linear(input))
        a_it = torch.softmax(self.u_w(u_it), dim=1)
        s_i = input * a_it
        return s_i


class BaseSelfAttention(nn.Module):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8421023
    def __init__(
            self,
            d_hid: int = 64
    ):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid, 1)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(1, 1)

    def forward(
            self,
            x: Tensor,
            val_l=None
    ):
        att = self.att_pool(self.att_fc1(x))
        att = self.att_fc2(att).squeeze(-1)
        if val_l is not None:
            for idx in range(len(val_l)):
                att[idx, val_l[idx]:] = -1e6
        att = torch.softmax(att, dim=1)
        x = (att.unsqueeze(2) * x).sum(axis=1)
        return x


class FuseBaseSelfAttention(nn.Module):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8421023
    def __init__(
            self,
            d_hid: int = 64,
            d_head: int = 4
    ):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid, 512)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(512, d_head)

        self.d_hid = d_hid
        self.d_head = d_head

    def forward(
            self,
            x: Tensor,
            val_a=None,
            val_b=None,
            a_len=None
    ):
        att = self.att_pool(self.att_fc1(x))
        # att = self.att_fc2(att).squeeze(-1)
        att = self.att_fc2(att)
        att = att.transpose(1, 2)
        if val_a is not None:
            for idx in range(len(val_a)):
                att[idx, :, val_a[idx]:a_len] = -1e5
                att[idx, :, a_len + val_b[idx]:] = -1e5
        att = torch.softmax(att, dim=2)
        # x = torch.matmul(att, x).mean(axis=1)
        x = torch.matmul(att, x)
        x = x.reshape(x.shape[0], self.d_head * self.d_hid)
        return x