'''
Copyright (C) 2021 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
(Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
'''
#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Copy-paste from SparseRCNN Transformer class with modifications:
    * no bbox input, pooler input, bbox prediction part
"""
import copy
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from ..registry import HEADS
from ..utils import ConvModule

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)
_SOFTMAX_MASKING_CONSTANT = -99999.0

from timm.models.layers import DropPath
from einops import rearrange


@HEADS.register_module
class MultiScaleDynamicMaskHead(nn.Module):

    def __init__(self, dh_dim=256, num_classes=9, dim_feedforward=2048, nhead=8,
                 dropout=0.0, activation="relu",
                 dh_num_heads=8,
                 per_dh_num_heads=2,  # [2, 2, 2, 2]
                 feat_num_levels=4,
                 merge_operation="add",   # "concat"
                 trans_in_dim=128,
                 return_intermediate=True,
                 use_focal=True,
                 prior_prob=0.01,
                 num_cls=1, num_reg=3,
                 softmax_dim="slots",
                 drop_path=0.,
                 temporal_query_attention_config=None,
                 apply_temporal_query_atten_stages=None,
                 other_config=None,
                 ):
        super(MultiScaleDynamicMaskHead, self).__init__()

        if not isinstance(per_dh_num_heads, list):
            assert per_dh_num_heads * feat_num_levels == dh_num_heads
            per_dh_num_heads = [per_dh_num_heads] * feat_num_levels
        else:
            assert sum(per_dh_num_heads) == dh_num_heads
        self.dh_num_heads_accu = [per_dh_num_heads[0]]
        for ti in range(1, len(per_dh_num_heads)):
            self.dh_num_heads_accu.append(self.dh_num_heads_accu[-1] + per_dh_num_heads[ti])

        self.dh_dim = dh_dim
        self.trans_in_dim = trans_in_dim
        self.apply_temporal_query_atten_stages = apply_temporal_query_atten_stages
        self.other_config = other_config

        # Build heads.
        rcnn_head = MaskRCNNHead(d_model=dh_dim, num_classes=num_classes,
                                 dim_feedforward=dim_feedforward, nhead=nhead,
                                 dropout=dropout, activation=activation,
                                 num_cls=num_cls, num_reg=num_reg,
                                 softmax_dim=softmax_dim,
                                 drop_path=drop_path,
                                 temporal_query_attention_config=temporal_query_attention_config,
                                 )
        all_stage_num = 0
        # separately build heads for different feature levels.
        for i in range(feat_num_levels):
            if apply_temporal_query_atten_stages is None:
                # default
                setattr(self, "head_series_{}".format(i),
                        _get_clones(rcnn_head, N=per_dh_num_heads[i])
                        )
            else:
                if all_stage_num in apply_temporal_query_atten_stages:
                    setattr(self, "head_series_{}".format(i),
                            _get_clones(rcnn_head, N=per_dh_num_heads[i])
                            )
                else:
                    pure_rcnn_head = MaskRCNNHead(d_model=dh_dim, num_classes=num_classes,
                                                  dim_feedforward=dim_feedforward, nhead=nhead,
                                                  dropout=dropout, activation=activation,
                                                  num_cls=num_cls, num_reg=num_reg,
                                                  softmax_dim=softmax_dim,
                                                  drop_path=drop_path,
                                                  temporal_query_attention_config=None,  # no temporal attention.
                                                  )
                    setattr(self, "head_series_{}".format(i),
                            _get_clones(pure_rcnn_head, N=per_dh_num_heads[i])
                            )

            all_stage_num += per_dh_num_heads[i]

        # each time after upsampling, still need conv to transform features before querying.
        # 128, 128 for add, 256, 128 or 256, 256 for concat
        self.conv_trans = ConvModule(trans_in_dim,
                                     dh_dim, 1,
                                     padding=0,
                                     activation=None)

        self.return_intermediate = return_intermediate  # default True
        self.feat_num_levels = feat_num_levels
        self.merge_operation = merge_operation

        # Init parameters.
        self.use_focal = use_focal
        self.num_classes = num_classes
        if self.use_focal:
            self.prior_prob = prior_prob
            self.bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    nn.init.constant_(p, self.bias_value)

    def forward(self, features, init_masks, pad_mask, pos=None, query_pos=None, gt_non_void_mask=None):

        inter_class_logits = []
        inter_pred_masks = []
        if len(inter_class_logits) == 0:
            for _ in range(len(features)):
                inter_class_logits.append([])
                inter_pred_masks.append([])
        all_stage_index = 0

        bs = len(features[0][0])

        mask_queries = []
        for i in range(len(features)):
            init_masks[i] = init_masks[i][None].repeat(bs, 1, 1)  # (1, bs, 1)
            mask_queries.append(init_masks[i])  # .clone()
            # the query_pos is for position embedding of mask_query
            if query_pos is not None:
                query_pos[i] = query_pos[i][None].repeat(bs, 1, 1)  # (1, bs, 1)

        # cancat features along frame dimension.
        update_features = []
        for si in range(len(features[0])):
            update_features_si = []
            for f_index in range(len(features)):
                update_features_si.append(features[f_index][si])
            update_features.append(torch.cat(update_features_si, dim=0))

        for i in range(self.feat_num_levels):
            assert gt_non_void_mask is None
            # so do not need to handle gt_non_void_mask
            each_non_void_mask = gt_non_void_mask[i] if gt_non_void_mask is not None else None
            # first fuse, and transform then query, upsample
            curr_feat = update_features[i]
            if i > 0:
                # fuse features of different levels
                if self.merge_operation == 'add':
                    curr_feat = curr_feat + F.interpolate(update_features[i - 1], None, 2, mode='bilinear',
                                                          align_corners=False)
                elif self.merge_operation == 'concat':
                    curr_feat = torch.cat((F.interpolate(update_features[i - 1], None, 2, mode='bilinear',
                                                         align_corners=False), curr_feat), dim=1)
                # further transform features
                curr_feat = self.conv_trans(curr_feat)
            if i == 0 and self.dh_dim != curr_feat.size(1) and self.trans_in_dim == curr_feat.size(1) * 3:
                curr_feat = torch.cat((curr_feat, curr_feat, curr_feat), dim=1)
                # further transform features
                curr_feat = self.conv_trans(curr_feat)

            # split the concatenated features into list.
            curr_feat = torch.split(curr_feat, bs, dim=0)

            # query masks
            for rcnn_head in getattr(self, "head_series_{}".format(i)):
                curr_pos = None if pos is None else [p[i] for p in pos]  # pos[i]
                assert pad_mask is None
                assert query_pos is None
                assert each_non_void_mask is None
                stage_enable = False
                if all_stage_index in self.apply_temporal_query_atten_stages:
                    stage_enable = True
                class_logits, pred_masks, curr_saved_mask_query, ffn_out = \
                    rcnn_head(curr_feat, mask_queries,
                              pad_mask, curr_pos,
                              query_pos,
                              gt_non_void_mask=each_non_void_mask,
                              stage_enable=stage_enable)
                if self.return_intermediate:
                    assert len(class_logits) == len(features), "{}, {}".format(len(class_logits), len(features))
                    for ti in range(len(class_logits)):
                        inter_class_logits[ti].append(class_logits[ti])
                        inter_pred_masks[ti].append(pred_masks[ti])
                for ti in range(len(class_logits)):
                    mask_queries[ti] = pred_masks[ti].detach()
                all_stage_index += 1

            # to keep the resolution of each level, move upsample to next stage
            update_features[i] = torch.cat(curr_feat, dim=0)

        # change the update_features dimension.
        update_features = [torch.split(uf, bs, dim=0) for uf in update_features]
        return_features = []
        for f_i in range(len(update_features[0])):
            return_features.append([update_features[s_i][f_i] for s_i in range(len(update_features))])

        if self.return_intermediate:
            return [torch.stack(icl) for icl in inter_class_logits], \
                   [torch.stack(ipm) for ipm in inter_pred_masks], \
                   return_features

        return [cl[None] for cl in class_logits], [pm[None] for pm in pred_masks], None


class MaskRCNNHead(nn.Module):

    def __init__(self, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP,
                 num_cls=1, num_reg=3, use_focal=True,
                 softmax_dim="slots", drop_path=0.,
                 temporal_query_attention_config=None):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = MaskDynamicConv(dh_dim=d_model, softmax_dim=softmax_dim)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # DropPath
        self.drop_path = None
        if drop_path > 0:
            self.drop_path = DropPath(drop_path)

        self.temporal_query_head = None
        if temporal_query_attention_config is not None:
            self.temporal_query_head = TemporalSlotsHead(**temporal_query_attention_config)

        # cls.
        # num_cls = cfg.MODEL.SparseRCNN.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        # although the name is reg_module, but actually is only for convert the query before next stage.
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.use_focal = use_focal
        self.class_logits = nn.Linear(d_model, num_classes)
        self.scale_clamp = scale_clamp

    def forward(self, features, mask_query, pad_mask, pos=None, query_pos=None,
                gt_non_void_mask=None, stage_enable=True,):
        """
        :param bboxes: (N, nr_boxes, 4) - > (nr_boxes, 4)
        :param mask_query: (N, nr_boxes, d_model)  1, 512, 256
        :param stage_enable use when forward_multi_frames
        """
        N, nr_boxes = mask_query[0].shape[:2]
        query_dim = mask_query[0].shape[2]
        multi_obj_features = []
        fc_features = []  # for return
        for i in range(len(features)):
            obj_features_i = self.forward_till_ffn(features[i], mask_query[i], pos[i], query_pos,
                                                   gt_non_void_mask=gt_non_void_mask)
            multi_obj_features.append(obj_features_i)
            fc_features.append(obj_features_i)
            # [1, 100, 256]
        if stage_enable:
            # concat along the slot dimension.
            multi_obj_features = torch.cat(multi_obj_features, dim=1)  # 1, 200, 256

            # do the Video Retriever.
            multi_obj_features_refine = self.temporal_query_head(features=multi_obj_features[0],
                                                                 mask_query=multi_obj_features[0],
                                                                 pos=None, query_pos=None)
            multi_obj_features_refine = multi_obj_features_refine.unsqueeze(0)
            multi_obj_features_refine = multi_obj_features + multi_obj_features_refine

            # distribute to each frame
            multi_obj_features_refine = torch.split(multi_obj_features_refine,
                                                    multi_obj_features_refine.size(1) // len(mask_query),
                                                    dim=1)  # [1, 100, 256] * 2

            all_class_logits, all_reg_features = [], []
            for mi in range(len(multi_obj_features_refine)):
                each_class_logits, each_reg_feature, _ = self.forward_after_ffn(
                    multi_obj_features_refine[mi], N, nr_boxes)
                all_class_logits.append(each_class_logits)
                all_reg_features.append(each_reg_feature)
        else:
            # do not forward the temporal_query_head.
            assert self.temporal_query_head is None
            all_class_logits, all_reg_features = [], []
            for mi in range(len(multi_obj_features)):
                each_class_logits, each_reg_feature, _ = self.forward_after_ffn(
                    multi_obj_features[mi], N, nr_boxes)
                all_class_logits.append(each_class_logits)
                all_reg_features.append(each_reg_feature)

        return all_class_logits, all_reg_features, None, fc_features

    def forward_till_ffn(self, features, mask_query, pos=None, query_pos=None, gt_non_void_mask=None):
        N, nr_boxes = mask_query.shape[:2]

        # self_att.
        pro_features = mask_query.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        if query_pos is not None:
            pro_features2 = self.self_attn(with_pos_embed(pro_features, query_pos.permute(1, 0, 2)),
                                           with_pos_embed(pro_features, query_pos.permute(1, 0, 2)),
                                           value=pro_features, key_padding_mask=None)[0]
        else:
            pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features, key_padding_mask=None)[0]
        if self.drop_path is not None:
            pro_features = pro_features + self.drop_path(pro_features2)
        else:
            pro_features = pro_features + self.dropout1(pro_features2)  # default

        pro_features = self.norm1(pro_features)
        # [100, 1, 256]

        # inst_interact.
        pro_features = pro_features.permute(1, 0, 2)
        if query_pos is not None:
            pro_features2 = self.inst_interact(with_pos_embed(pro_features, query_pos),
                                               features, pos, gt_non_void_mask=gt_non_void_mask)

        else:
            pro_features2 = self.inst_interact(pro_features, features, pos,
                                               gt_non_void_mask=gt_non_void_mask)

        if self.drop_path is not None:
            pro_features = pro_features + self.drop_path(pro_features2)
        else:
            pro_features = pro_features + self.dropout2(pro_features2)  # default

        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))

        if self.drop_path is not None:
            obj_features = obj_features + self.drop_path(obj_features2)
        else:
            obj_features = obj_features + self.dropout3(obj_features2)  # default
        obj_features = self.norm3(obj_features)
        # [1, 100, 256]

        return obj_features

    def forward_after_ffn(self, obj_features, N, nr_boxes):
        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)

        return class_logits.view(N, nr_boxes, -1), reg_feature.view(N, nr_boxes, -1), None


class MaskDynamicConv(nn.Module):
    # Retriever in the Panoptic Retriever.

    def __init__(self, dh_dim=256, softmax_dim="slots"):
        super().__init__()

        self.hidden_dim = dh_dim
        self.softmax_dim = softmax_dim

        self.to_q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.to_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.to_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

        self.norm_q = nn.LayerNorm(self.hidden_dim)
        self.norm_k = nn.LayerNorm(self.hidden_dim)
        self.norm_v = nn.LayerNorm(self.hidden_dim)

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, pro_features, features, pos, gt_non_void_mask=None):
        '''
        pro_features: (N, L, C)
        features: (N, H, W, C)
        '''
        features = rearrange(features, "b c h w -> b h w c")
        if pos is not None:
            pos = rearrange(pos, "b c h w -> b h w c")
        q = self.norm_q(self.to_q(pro_features))
        k = self.norm_k(self.to_k(with_pos_embed(features, pos)))
        v = self.norm_v(self.to_v(features))

        attn = torch.einsum("b l c, b h w c -> b l h w", q, k)
        if gt_non_void_mask is not None:
            if gt_non_void_mask.size() != attn.size():
                gt_non_void_mask = gt_non_void_mask.view(1, 1, gt_non_void_mask.size(0), gt_non_void_mask.size(1)) \
                    .repeat(attn.size(0), attn.size(1),
                            1, 1)
            if gt_non_void_mask.dtype == torch.bool:
                attn = attn.masked_fill(~gt_non_void_mask, value=float('-inf'))
            else:   # default
                attn = attn.masked_fill((1 - gt_non_void_mask).bool(), value=float('-inf'))
        if self.softmax_dim == "slots":
            attn = F.softmax(attn, dim=1)
        elif self.softmax_dim == "hw":
            attn = F.softmax(attn.flatten(2), dim=-1).view(attn.size())
        else:
            assert 1 == 2, "self.softmax_dim is not VALID !!!!!!"
        if gt_non_void_mask is not None:
            if gt_non_void_mask.dtype == torch.bool:
                attn = attn.masked_fill(~gt_non_void_mask, value=0)
            else:   # default
                attn = attn.masked_fill((1 - gt_non_void_mask).bool(), value=0)
        out = torch.einsum("b l h w, b h w c -> b l c", attn, v)

        out = self.norm1(out)
        out = self.activation(out)

        return out


@HEADS.register_module
class TemporalSlotsHead(nn.Module):
    # Video Retriever.

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, activation="relu",
                 softmax_dim="slots", drop_path=0.):
        super().__init__()

        self.d_model = d_model

        self.inst_interact = SlotsDynamicConv(dh_dim=d_model, softmax_dim=softmax_dim)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # DropPath
        self.drop_path = None
        if drop_path > 0:
            self.drop_path = DropPath(drop_path)

    def forward(self, features, mask_query, pos=None, query_pos=None):
        """
        :param features: (N, nr_boxes, d_model) -> slots in previous frames
        :param mask_query: (N, nr_boxes, d_model)  1, 512, 256
        """
        nr_boxes = mask_query.shape[0]
        prev_nr_boxes = features.shape[0]

        # self_att.
        pro_features = mask_query.view(1, nr_boxes, self.d_model)
        features = features.view(1, prev_nr_boxes, self.d_model)

        # the interaction is among queries.
        if query_pos is not None:
            pro_features2 = self.inst_interact(with_pos_embed(pro_features, query_pos),
                                               features, pos)
        else:
            pro_features2 = self.inst_interact(pro_features, features, pos)
        if self.drop_path is not None:
            pro_features = pro_features + self.drop_path(pro_features2)
        else:
            pro_features = pro_features + self.dropout2(pro_features2)  # default

        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        if self.drop_path is not None:
            obj_features = obj_features + self.drop_path(obj_features2)
        else:
            obj_features = obj_features + self.dropout3(obj_features2)  # default
        obj_features = self.norm3(obj_features)

        return obj_features.squeeze(0)


class SlotsDynamicConv(nn.Module):
    # Retriever of Video Retriever. Difference mainly lies in the inputs.

    def __init__(self, dh_dim=256, softmax_dim="slots"):
        super().__init__()

        self.hidden_dim = dh_dim
        self.softmax_dim = softmax_dim

        self.to_q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.to_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.to_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

        self.norm_q = nn.LayerNorm(self.hidden_dim)
        self.norm_k = nn.LayerNorm(self.hidden_dim)
        self.norm_v = nn.LayerNorm(self.hidden_dim)

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, curr_features, features, pos):
        '''
        curr_features: (N, L, C) -> the slots in current frame
        prev_features: (N, L, C) -> the slots in previous frames.
        '''
        q = self.norm_q(self.to_q(curr_features))
        k = self.norm_k(self.to_k(with_pos_embed(features, pos)))  # pos is actually None
        v = self.norm_v(self.to_v(features))

        attn = torch.einsum("b l c, b u c -> b l u", q, k)
        # print("attn: {}".format(attn.size()))  # 1, 100, 100
        if self.softmax_dim == "slots":
            attn = F.softmax(attn, dim=1)
        elif self.softmax_dim == "ref_slots":
            attn = F.softmax(attn, dim=2)
        else:
            assert 1 == 2, "self.softmax_dim is not VALID !!!!!!"
        out = torch.einsum("b l u, b u c -> b l c", attn, v)

        out = self.norm1(out)
        out = self.activation(out)

        return out


def with_pos_embed(tensor, pos, h=None, w=None):
    if pos is None:
        return tensor
    if len(pos.shape) > len(tensor.shape):
        assert 1 == 2, "pos shape: {}, tensor shape: {}".format(pos.shape, tensor.shape)
        tensor = tensor.permute(1, 2, 0)
        tensor = tensor.contiguous().view(tensor.size(0), tensor.size(1), h, w)
    tensor = tensor + pos
    return tensor.contiguous()


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

