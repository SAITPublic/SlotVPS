'''
Copyright (C) 2021 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
(Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
'''

import torch.nn as nn
from ..registry import DETECTORS

from mmcv.cnn import kaiming_init, xavier_init
from .position_encoding import build_position_encoding

from .. import builder
from .dynamic_mask_head import MultiScaleDynamicMaskHead
from ..utils.conv_module import init_weights as general_init_weights
from ..utils import ConvModule

_SOFTMAX_MASKING_CONSTANT = -99999.0


@DETECTORS.register_module
class VPS_Capsule(nn.Module):
    # the image model.

    def __init__(self,
                 backbone,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 panoptic=None,
                 dynamic_mask_head=None,
                 pretrained=None,
                 other_config=None):
        super(VPS_Capsule, self).__init__()
        self.fp16_enabled = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.other_config = other_config
        self.cfg = self.train_cfg if self.train_cfg is not None else self.test_cfg
        self.has_no_obj = self.other_config.get('has_no_obj', True)
        self.num_classes = dynamic_mask_head['num_classes']
        if self.num_classes <= 20:
            # means the Cityscapes related dataset
            self.stuff_num = 11
        elif self.num_classes == 46 or self.num_classes == 47:
            # means the MV dataset.
            self.stuff_num = 34
        elif self.num_classes == 23 or self.num_classes == 24:
            # means the viper dataset
            self.stuff_num = 13
        else:
            assert 1 == 2, "self.num_classes: {}".format(self.num_classes)
        if self.has_no_obj:
            assert dynamic_mask_head['num_classes'] in [9, 20, 47, 24]

        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        if panoptic is not None:
            if "feat_num_levels" in dynamic_mask_head:
                panoptic['return_feat_levels'] = dynamic_mask_head['feat_num_levels']
            self.panopticFPN = builder.build_panoptic(panoptic)

        if dynamic_mask_head is not None:
            self.init_mask_query = nn.Embedding(other_config.get('proposal_num', 100), dynamic_mask_head['dh_dim'])
            xavier_init(self.init_mask_query)

            panoptic_out_channel = panoptic['out_channels']
            main_trans_out_dim = self.other_config.get("main_trans_out_dim", panoptic['out_channels'])
            self.conv_trans = ConvModule(panoptic_out_channel,
                                         main_trans_out_dim, 1,
                                         padding=0,
                                         activation=None)

            dynamic_mask_head['other_config'] = other_config
            self.dynamic_mask_head = MultiScaleDynamicMaskHead(**dynamic_mask_head)
            self.query_feat_num_levels = dynamic_mask_head.get("feat_num_levels", 4)
            self.multi_scale_heads_num = dynamic_mask_head.get("per_dh_num_heads", [1, 2, 2, 2])
            # make sure that aux_mask_outputs of multi-scale mask head use corresponding feature maps.
            if self.other_config.get("matched_feat_version", 0) == 1:
                self.matched_feat_indexes = []
                temp_ii = 0
                for mshn in self.multi_scale_heads_num:
                    self.matched_feat_indexes += [temp_ii] * mshn
                    temp_ii = temp_ii + 1
                # print("self.matched_feat_indexes: {}".format(self.matched_feat_indexes))

        self.position_embedding = build_position_encoding(self.other_config.get("pos_config", None))

        self.fg_bn = nn.BatchNorm2d(1)
        self.feat_bn = nn.BatchNorm2d(dynamic_mask_head['dh_dim'])

        self.upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True
        )

        self.init_weights(pretrained=pretrained)

        # for debugging
        self.iter = 0
        self.save_dir = other_config.get("output_dir", "")

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_panoptic(self):
        return hasattr(self, 'panopticFPN') and self.panopticFPN is not None

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_panoptic:
            self.panopticFPN.init_weights()
        general_init_weights(self.modules())

        self.fg_bn.weight.data.fill_(0.1)
        self.fg_bn.bias.data.zero_()

        self.feat_bn.weight.data.fill_(1)
        self.feat_bn.bias.data.zero_()