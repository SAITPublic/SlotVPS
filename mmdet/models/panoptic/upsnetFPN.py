import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..registry import PANOPTIC
from ..utils import DeformConvWithOffset, ConvModule

import torch

@PANOPTIC.register_module
class UPSNetFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_levels,
                 num_things_classes,
                 num_classes,
                 ignore_label,
                 loss_weight,
                 conv_cfg=None,
                 norm_cfg=None,
                 return_feat_levels=4,):
        super(UPSNetFPN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.num_things_classes = num_things_classes # 8
        self.num_classes = num_classes # 19
        self.num_stuff_classes = num_classes - num_things_classes # 11
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.deform_convs = nn.ModuleList()
        self.deform_convs.append(nn.Sequential( 
            DeformConvWithOffset(self.in_channels, self.in_channels, 
                                 kernel_size=3, padding=1),
            nn.GroupNorm(32, self.in_channels),
            nn.ReLU(inplace=True),
            DeformConvWithOffset(self.in_channels, self.out_channels,
                                 kernel_size=3, padding=1),
            nn.GroupNorm(32, self.out_channels),
            nn.ReLU(inplace=True),
            DeformConvWithOffset(self.out_channels, self.out_channels,
                                 kernel_size=3, padding=1),
            nn.GroupNorm(32, self.out_channels),
            nn.ReLU(inplace=True),            
            ))

        self.return_feat_levels = return_feat_levels

        self.conv_pred = ConvModule(self.out_channels * 4,
                                    self.num_classes, 1,
                                    padding=0,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    activation=None)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.num_levels
        fpn_px = []
        for i in range(self.num_levels):
            fpn_px.append(self.deform_convs[0](inputs[i]))

        if self.return_feat_levels == 4:
            feat_before = [fpn_px[3], fpn_px[2], fpn_px[1], fpn_px[0]]
        elif self.return_feat_levels == 3:
            feat_before = [fpn_px[2], fpn_px[1], fpn_px[0]]

        fpn_p2 = fpn_px[0]
        fpn_p3 = F.interpolate(fpn_px[1], None, 2, mode='bilinear', align_corners=False)
        fpn_p4 = F.interpolate(fpn_px[2], None, 4, mode='bilinear', align_corners=False)
        fpn_p5 = F.interpolate(fpn_px[3], None, 8, mode='bilinear', align_corners=False)
        feat = torch.cat([fpn_p2, fpn_p3, fpn_p4, fpn_p5], dim=1)

        fcn_score = self.conv_pred(feat)
        fcn_output = self.upsample(fcn_score)
        return fcn_output, fcn_score, feat_before

    def loss(self, segm_pred, segm_label):
        loss = dict()
        if isinstance(self.loss_weight, list):
            assert isinstance(segm_pred, list)
            assert len(self.loss_weight) == len(segm_pred)
            loss['loss_segm'] = sum(
                [w * F.cross_entropy(x, segm_label, ignore_index=self.ignore_label) for (w, x) in
                 zip(self.loss_weight, segm_pred)])
        else:
            loss_segm = F.cross_entropy(segm_pred, segm_label, ignore_index = self.ignore_label)
            loss['loss_segm'] = self.loss_weight * loss_segm
        return loss