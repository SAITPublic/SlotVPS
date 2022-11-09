'''
Copyright (C) 2021 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
(Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
'''
# -------------------------------------------------
# Modified based on:
# Video Instance Segmentation
# (https://github.com/youtubevos/MaskTrackRCNN/)
#---------------------------------------------------
import torch
import torch.nn as nn
from ..registry import HEADS

@HEADS.register_module
class SimpleTrackHead(nn.Module):
    """Tracking head, predict tracking features and match with reference objects
       Use dynamic option to deal with different number of objects in different
       images. A non-match entry is added to the reference objects with all-zero
       features. Object matched with the non-match entry is considered as a new
       object.
    """

    def __init__(self,
                 num_fcs_query=0,
                 in_channels_query=0,
                 loss_match=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                 query_matched_weight=1.0,
                 ):

        super(SimpleTrackHead, self).__init__()
        self.query_matched_weight = query_matched_weight
        self.num_fcs_query = num_fcs_query

        if num_fcs_query > 0:
            self.fcs_query = nn.ModuleList()
            for i in range(num_fcs_query):
                fc_query = nn.Linear(in_channels_query, in_channels_query)
                self.fcs_query.append(fc_query)
            self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self):
        if self.num_fcs_query > 0:
            for fc in self.fcs_query:
                nn.init.normal_(fc.weight, 0, 0.01)
                nn.init.constant_(fc.bias, 0)

    def forward(self, x_query=None, ref_x_query=None):
        # here we compute a correlation matrix of x_query and ref_x_query.
        # we also add a all 0 column denote no matching.

        # if "pred_masks_output_query"
        if self.num_fcs_query > 0:
            # forward fcs
            for idx, fc in enumerate(self.fcs_query):
                x_query = fc(x_query)
                if idx < len(self.fcs_query) - 1:
                    x_query = self.relu(x_query)

                if isinstance(ref_x_query, list):
                    for i in range(len(ref_x_query)):
                        ref_x_query[i] = fc(ref_x_query[i])
                        if idx < len(self.fcs_query) - 1:
                            ref_x_query[i] = self.relu(ref_x_query[i])
                else:
                    ref_x_query = fc(ref_x_query)
                    if idx < len(self.fcs_query) - 1:
                        ref_x_query = self.relu(ref_x_query)

        # calculate correlation matrix
        if ref_x_query is not None and isinstance(ref_x_query, list):
            loop_len = len(ref_x_query)
            match_score = []
            for i in range(loop_len):
                prod_i = torch.mm(x_query, torch.transpose(ref_x_query[i], 0, 1))
                dummy_i = torch.zeros(prod_i.size(0), 1, device=torch.cuda.current_device(), dtype=prod_i.dtype)
                match_score.append(torch.cat([dummy_i, prod_i], dim=1))
        else:
            prod = torch.mm(x_query, torch.transpose(ref_x_query, 0, 1))
            dummy = torch.zeros(prod.size(0), 1, device=torch.cuda.current_device(), dtype=prod.dtype)
            match_score = [torch.cat([dummy, prod], dim=1)]

        return match_score