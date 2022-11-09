'''
Copyright (C) 2021 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
(Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
from mmdet.core import auto_fp16
from ..registry import DETECTORS

import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.core.utils.misc import NestedTensor, nested_tensor_from_tensor_list
from .vps_capsule import VPS_Capsule
from .simple_track_head import SimpleTrackHead
from ..structures import Instances
from ..utils.conv_module import init_weights as general_init_weights
from mmdet.core.utils.misc import interpolate
from collections import defaultdict
from PIL import Image
from panopticapi.utils import rgb2id, id2rgb
from einops import rearrange

_SOFTMAX_MASKING_CONSTANT = -99999.0

def construct_instance(src):
    res = Instances((1, 1))
    res.output_embedding = src.output_embedding
    if src.has("saved_mask_query"):
        res.saved_mask_query = src.saved_mask_query
    return res

@DETECTORS.register_module
class VPS_Temporal_Slots(nn.Module):

    def __init__(self,
                 backbone,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 panoptic=None,
                 dynamic_mask_head=None,
                 pretrained=None,
                 postprocess_panoptic=None,
                 simple_track_head=None,
                 other_config=None):
        super(VPS_Temporal_Slots, self).__init__()

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

        # build image model
        self.image_model = VPS_Capsule(backbone,
                                       train_cfg,
                                       test_cfg,
                                       neck=neck,
                                       panoptic=panoptic,
                                       dynamic_mask_head=dynamic_mask_head,
                                       pretrained=pretrained,
                                       other_config=other_config)

        self.track_head_config = simple_track_head
        if simple_track_head is not None:
            self.temporal_track_head = SimpleTrackHead(**simple_track_head)

        # self.track_base = RuntimeTrackerBase()
        if postprocess_panoptic is not None:
            self.postprocess_panoptic = PostProcessPanopticInstances(**postprocess_panoptic)

        self.init_weights()

        # for debugging
        self.iter = 0
        self.save_dir = other_config.get("output_dir", "")

    def init_weights(self):
        general_init_weights(self.modules())

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.image_model.init_mask_query.weight.shape  # (300, 512)
        device = self.image_model.init_mask_query.weight.device
        track_instances.init_masks_query = self.image_model.init_mask_query.weight
        # the output mask query weight
        track_instances.output_embedding = torch.zeros((num_queries, dim), device=device)
        # the object index
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        # use in test stage, for helping disappear and reappear
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)
        # should be the track score
        # track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        # new added, the predicted track logits from the new branch
        # the predicted class scores
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)

        return track_instances.to(device)

    def extract_semantic_feats(self, x):
        fcn_output, fcn_score, fcn_feature = \
            self.image_model.panopticFPN(x[0:self.image_model.panopticFPN.num_levels])  # 0:4
        if fcn_output.dtype == torch.float16:
            fcn_output = fcn_output.type(torch.float32)
        return fcn_output, fcn_score, fcn_feature

    def semantic_trans_ins(self, fcn_feature):
        # use multiple scale dynamic mask head, and the query features are from semantic head
        feature_trans = []
        assert len(fcn_feature) == self.image_model.query_feat_num_levels
        for ii in range(len(fcn_feature)):
            feature_trans.append(self.image_model.conv_trans(fcn_feature[ii]))
        return feature_trans

    def generate_position_embedding(self, dh_head_input_feats):
        # need to generate multi-scale postition embedding
        pos_embed = []
        for each_level_feat in dh_head_input_feats:
            pos_embed.append(self.image_model.position_embedding(nested_tensor_from_tensor_list((each_level_feat))))
        return pos_embed

    def generate_final_outputs(self, dh_head_input_feats, outputs_masks, generate_aux_output=True):
        for ii in range(len(dh_head_input_feats)):
            dh_head_input_feats[ii] = self.image_model.feat_bn(dh_head_input_feats[ii])
            dh_head_input_feats[ii] = F.normalize(dh_head_input_feats[ii], p=2, dim=1)

        mask_output = torch.einsum("n c h w, n l c -> n l h w", dh_head_input_feats[-1], outputs_masks[-1])

        if mask_output.size(0) == 1:
            # default batch size = 1
            mask_output = self.image_model.fg_bn(rearrange(mask_output, "b l h w -> l b h w"))
            mask_output = rearrange(mask_output, "l b h w -> b l h w")
        else:
            for bi in range(mask_output.size(0)):
                mask_output[bi] = rearrange(
                    self.image_model.fg_bn(rearrange(mask_output[bi].unsqueeze(0), "b l h w -> l b h w")),
                    "l b h w -> b l h w")[0]

        aux_mask_outputs = []
        if generate_aux_output:
            # if len(outputs_class) > 1:
            for i in range(len(outputs_masks) - 1):
                if self.other_config.get("matched_feat_version", 0) == 0:
                    feat_index = i // 2
                elif self.other_config.get("matched_feat_version", 0) == 1:
                    feat_index = self.image_model.matched_feat_indexes[i]
                elif self.other_config.get("matched_feat_version", 0) == 2:
                    # all utilize the biggest feature map.
                    feat_index = -1
                else:
                    assert 1 == 2
                # use corresponding feature map.
                aux_mask_output = torch.einsum("n c h w, n l c -> n l h w", dh_head_input_feats[feat_index],
                                               outputs_masks[i])
                aux_mask_output = F.interpolate(aux_mask_output, None,
                                                mask_output.size(-1) // aux_mask_output.size(-1),
                                                mode='bilinear', align_corners=False)

                if aux_mask_output.size(0) == 1:
                    # default batch size = 1
                    aux_mask_output = self.image_model.fg_bn(rearrange(aux_mask_output, "b l h w -> l b h w"))
                    aux_mask_output = rearrange(aux_mask_output, "l b h w -> b l h w")
                else:
                    for bi in range(aux_mask_output.size(0)):
                        aux_mask_output[bi] = rearrange(
                            self.image_model.fg_bn(
                                rearrange(aux_mask_output[bi].unsqueeze(0), "b l h w -> l b h w")),
                            "l b h w -> b l h w")[0]

                aux_mask_outputs.append(aux_mask_output)

        return dh_head_input_feats, mask_output, aux_mask_outputs

    def generate_output_dict_test(self, outputs_class, mask_output, outputs_masks, fcn_output,
                             dh_head_input_feats):
        outputs = {'pred_logits': outputs_class[-1], 'pred_masks': mask_output,
                   'pred_masks_embed': outputs_masks[-1]}
        outputs['aux_outputs'] = [{'pred_masks_embed': c}
                                  for c in
                                  outputs_masks[:-1]]
        outputs['fcn_output'] = fcn_output
        outputs['dh_head_features'] = dh_head_input_feats[-1]
        return outputs

    @auto_fp16(apply_to=('img', 'ref_img',))
    def simple_test(self, img, img_meta, rescale=False, ref_img=None):
        """Test without augmentation."""
        if ref_img is not None:
            ref_img = ref_img[0]

        if isinstance(img_meta, DC):
            img_meta = img_meta.data[0][0]
        else:
            img_meta = img_meta[0]

        iid = img_meta['iid']
        div_mod = 10000
        if self.num_classes == 23 or self.num_classes == 24:
            # means the viper dataset
            div_mod = 100000
        self.vid = iid // div_mod  # 10000
        self.fid = iid % div_mod  # 10000
        self.img_filename = img_meta['filename']

        is_first = (self.fid == 1)

        pano_results = None

        with torch.no_grad():
            if is_first:
                self.test_track_instances = self._generate_empty_tracks()
                # self.track_base.clear()
                # new added for use the track head to predict obj ids
                self.prev_instances = None
                self.prev_saved_querys = None
            else:
                assert self.test_track_instances is not None
                # self.test_track_instances.remove("labels")
                # self.test_track_instances.remove("masks")
                # self.test_track_instances.remove("probs")

            # extract_feat
            x = self.image_model.backbone(img)
            if self.other_config.get("test_forward_ref_img", False):
                ref_x = self.image_model.backbone(ref_img)

            if self.image_model.with_neck:
                x = self.image_model.neck(x)
                if self.other_config.get("test_forward_ref_img", False):
                    ref_x = self.image_model.neck(ref_x)

            # **********************************
            # FCN Semantic Head forward
            # **********************************
            if hasattr(self.image_model, 'panopticFPN') and self.image_model.panopticFPN is not None:
                fcn_output, fcn_score, fcn_feature = self.extract_semantic_feats(x)
                if self.other_config.get("test_forward_ref_img", False):
                    ref_fcn_output, ref_fcn_score, ref_fcn_feature = self.extract_semantic_feats(ref_x)

            # **********************************
            # Slot Attentions forward
            # **********************************
            if hasattr(self.image_model, 'dynamic_mask_head'):
                feature_trans = self.semantic_trans_ins(fcn_feature)
                if self.other_config.get("test_forward_ref_img", False):
                    ref_feature_trans = self.semantic_trans_ins(ref_fcn_feature)

                init_masks = self.test_track_instances.init_masks_query

                dh_head_input_feats = feature_trans
                if self.other_config.get("test_forward_ref_img", False):
                    ref_dh_head_input_feats = ref_feature_trans

                # generate position embedding
                pos_embed, ref_pos_embed = None, None
                pos_embed = self.generate_position_embedding(dh_head_input_feats)
                if self.other_config.get("test_forward_ref_img", False):
                    ref_pos_embed = self.generate_position_embedding(ref_dh_head_input_feats)

                multi_scale_pad_mask = None
                assert self.other_config.get("test_forward_ref_img", False) is True
                all_outputs_class, all_outputs_masks, all_dh_head_input_feats = self.image_model.dynamic_mask_head(
                    features=[ref_dh_head_input_feats, dh_head_input_feats],
                    init_masks=[init_masks, init_masks],
                    pad_mask=multi_scale_pad_mask,  # actually None
                    pos=[ref_pos_embed, pos_embed],
                    query_pos=None,
                    gt_non_void_mask=None,
                )
                ref_outputs_class, outputs_class = all_outputs_class
                ref_outputs_masks, outputs_masks = all_outputs_masks
                ref_dh_head_input_feats, dh_head_input_feats = all_dh_head_input_feats

                # generate final results
                dh_head_input_feats, mask_output, aux_mask_outputs = self.generate_final_outputs(dh_head_input_feats,
                                                                                                 outputs_masks,
                                                                                                 generate_aux_output=False)

                # prepare the output dict for calculating losses
                single_image_outputs = self.generate_output_dict_test(outputs_class, mask_output, outputs_masks,
                                                                      fcn_output,
                                                                      dh_head_input_feats)

            self.test_track_instances.pred_logits = single_image_outputs['pred_logits'][0]  # [100, 19]
            self.test_track_instances.pred_masks = single_image_outputs['pred_masks'][0]  # [100, 200, 400]
            self.test_track_instances.output_embedding = single_image_outputs['pred_masks_embed'][0]  # [100, 256]

            if self.num_classes in [19, 20]:
                assert img_meta['ori_shape'][0] == img.shape[2] and img_meta['ori_shape'][1] == int(img.shape[3]), \
                    "{}, {}".format(img_meta['ori_shape'], img.shape)
            if hasattr(self, 'postprocess_panoptic'):
                self.test_track_instances = self.postprocess_panoptic(self.test_track_instances,
                                                                      [(int(img_meta['ori_shape'][0]),
                                                                        int(img_meta['ori_shape'][1]))], id=iid)

            # self.track_base.update(self.test_track_instances)

            res_instances = self.test_track_instances.to(torch.device('cpu'))

            # convert the "stuff + instance" style into "instance" style
            panoptic_num = len(res_instances.labels)
            semantic_labels = res_instances.labels
            res_cls_inds = res_instances.labels
            ins_indexes = res_cls_inds > self.stuff_num - 1  # 10
            # need to -10 to align with the output of original panoptic_fusetrack
            res_cls_inds = res_cls_inds[ins_indexes] - (self.stuff_num - 1) # 10
            res_cls_prob = res_instances.probs[ins_indexes]
            track_res_instances = res_instances[:]

            if self.prev_instances is None:
                if self.other_config.get("test_only_save_main_results", False):
                    self.prev_instances = construct_instance(track_res_instances)
                else:
                    # default
                    self.prev_instances = track_res_instances
                res_det_obj_ids = torch.from_numpy(np.arange(panoptic_num))
                res_det_obj_ids = res_det_obj_ids[ins_indexes]
            else:
                # get match_score to update prev_instances.
                assert hasattr(self, 'temporal_track_head') and self.temporal_track_head is not None
                assert self.prev_instances is not None
                assert len(track_res_instances) > 0 and len(self.prev_instances) > 0
                match_score = self.temporal_track_head(track_res_instances.output_embedding.cuda(),
                                                       self.prev_instances.output_embedding.cuda())[0]
                match_logprob = torch.nn.functional.log_softmax(match_score, dim=1)

                # here directly use the match_logprob to serve as the comp_scores.
                match_likelihood, match_ids = torch.max(match_logprob, dim=1)
                # translate match_ids to det_obj_ids, assign new id to new objects
                # update tracking features/bboxes of exisiting object,
                # add tracking features/bboxes of new object
                match_likelihood = match_likelihood.cpu().numpy()
                match_ids = match_ids.cpu().numpy().astype(np.int32)  # [4]
                det_obj_ids = np.ones((match_ids.shape[0]), dtype=np.int32) * (-1)  # [-1, -1, -1, -1]
                best_match_scores = np.ones((len(self.prev_instances))) * (-100)  #
                best_match_ids = np.ones((len(self.prev_instances)), dtype=np.int32) * (
                    -1)  # [-100, -100, -100, -100, -100]

                for idx, match_id in enumerate(match_ids):
                    if match_id == 0:
                        # add new object
                        # NOTE. obj_ids is one of [0,1,...5],
                        # but is deducted by 1 later at "else".
                        # So, assign "5" makes sense.
                        det_obj_ids[idx] = len(self.prev_instances)  # [5]
                        if self.other_config.get("test_only_save_main_results", False):
                            self.prev_instances = Instances.cat(
                                [self.prev_instances, construct_instance(track_res_instances)[idx]])
                        else:
                            self.prev_instances = Instances.cat([self.prev_instances, track_res_instances[idx]])
                    else:
                        # multiple candidate might match with previous object, here we choose the one with
                        # largest comprehensive score
                        obj_id = match_id - 1
                        # match_score = comp_scores[idx, match_id]
                        match_score = match_likelihood[idx]
                        if match_score > best_match_scores[obj_id]:
                            det_obj_ids[idx] = obj_id
                            # if matched before, undo
                            if best_match_ids[obj_id] >= 0:
                                det_obj_ids[best_match_ids[obj_id]] = -1
                            best_match_scores[obj_id] = match_score
                            best_match_ids[obj_id] = idx
                            # udpate feature, currently directly replace the instance
                            if self.other_config.get("test_only_save_main_results", False):
                                self.prev_instances = Instances.cat([self.prev_instances[:obj_id],
                                                                     construct_instance(track_res_instances)[idx],
                                                                     self.prev_instances[obj_id + 1:]])
                            else:
                                self.prev_instances = Instances.cat(
                                    [self.prev_instances[:obj_id], track_res_instances[idx],
                                     self.prev_instances[obj_id + 1:]])

                # Above, objects that are considered redundant are marked "-1".
                # Here, we assign "NEW" label to all these objects.
                for idx, det_obj_id in enumerate(det_obj_ids):
                    if det_obj_id >= 0:
                        continue
                    det_obj_ids[idx] = len(self.prev_instances)  # self.prev_roi_feats.size(0)  # [5]
                    if self.other_config.get("test_only_save_main_results", False):
                        self.prev_instances = Instances.cat(
                            [self.prev_instances, construct_instance(track_res_instances)[idx]])
                    else:
                        self.prev_instances = Instances.cat([self.prev_instances, track_res_instances[idx]])

                res_det_obj_ids = torch.from_numpy(det_obj_ids)
                res_det_obj_ids = res_det_obj_ids[ins_indexes]

            instance_num = len(res_cls_inds)

            # mixture, need to reformulate the order.
            res_instances.masks = torch.cat((res_instances.masks[~ins_indexes], res_instances.masks[ins_indexes]), dim=0)
            semantic_labels = torch.cat((semantic_labels[~ins_indexes], semantic_labels[ins_indexes]), dim=0)

            panoptic_output = torch.max(
                F.softmax(res_instances.masks.unsqueeze(0), dim=1), dim=1)[1]
            # change the instance's indexes
            panoptic_ids = torch.unique(panoptic_output).long()
            instance_count = instance_num
            panoptic_output_2 = torch.zeros(panoptic_output.size(), dtype=panoptic_output.dtype,
                                            device=panoptic_output.device) * (-1)
            for i in range(len(panoptic_ids) - 1, -1, -1):
                object_id = panoptic_ids[i]
                region = (panoptic_output == object_id)
                if object_id >= (panoptic_num - instance_num):
                    # update the instance id.
                    panoptic_output_2[region] = self.stuff_num + instance_count - 1  # 11 +
                    instance_count = instance_count - 1
                else:
                    # update the semantic id.
                    panoptic_output_2[region] = semantic_labels[i]

            panoptic_output = panoptic_output_2

            if self.num_classes in [19, 20]:
                assert img_meta['ori_shape'] == img_meta['img_shape'], \
                    "{}, {}".format(img_meta['ori_shape'], img_meta['img_shape'])
            if single_image_outputs['fcn_output'].size(-2) != img_meta['ori_shape'][0] or \
                    single_image_outputs['fcn_output'].size(-1) != img_meta['ori_shape'][1]:
                input_img_h, input_img_w = img_meta['ori_shape'][:2]
                single_image_outputs['fcn_output'] = F.interpolate(single_image_outputs['fcn_output'],
                                                                   size=(int(input_img_h), int(input_img_w)),
                                                                   mode='bilinear',
                                                                   align_corners=False)
            fcn_output = torch.max(F.softmax(single_image_outputs['fcn_output'], dim=1), dim=1)[1]
            # trim out the padded boundaries --> original input shape
            img_shape_withoutpad = img_meta['ori_shape']  # img_shape
            fcn_output = fcn_output[:, 0:img_shape_withoutpad[0], 0:img_shape_withoutpad[1]]
            panoptic_output = panoptic_output[:, 0:img_shape_withoutpad[0], 0:img_shape_withoutpad[1]]

            temp_res = torch.unique(panoptic_output).long()
            if temp_res[temp_res > self.stuff_num - 1].size() != res_cls_inds.shape:
                print("MISMATCH !!!!!! res_classes: {}, res_cls_inds: {}, temp_res {}, {}, {}".format(semantic_labels,
                                                                                              res_cls_inds,
                                                                                              temp_res,
                                                                                              panoptic_num, instance_num))
            pano_results = {
                'fcn_outputs': fcn_output,
                'panoptic_cls_inds': res_cls_inds,
                'panoptic_cls_prob': res_cls_prob,
                'panoptic_det_obj_ids': res_det_obj_ids,
                'panoptic_outputs': panoptic_output,
            }

            self.test_track_instances = self._generate_empty_tracks()

        return pano_results

    def forward_test(self, imgs, img_metas, rescale=False, ref_img=None):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], rescale, ref_img)
        else:
            return self.aug_test(imgs, img_metas, rescale, ref_img)

    @auto_fp16(apply_to=('img', 'ref_img',))
    def forward(self, img, img_meta, return_loss=True,
                rescale=None,
                ref_img=None,  # images of reference frame
                ):  # **kwargs
        if return_loss:
            assert 1 == 2, "NOT RELEASED TRAIN CODE YET !!!!!!"
        else:
            return self.forward_test(img, img_meta, rescale, ref_img)


# class RuntimeTrackerBase(object):
#     def __init__(self, score_thresh=0.85, filter_score_thresh=0.6, miss_tolerance=5):  # score_thresh=0.7
#         self.score_thresh = score_thresh
#         self.filter_score_thresh = filter_score_thresh
#         self.miss_tolerance = miss_tolerance
#         self.max_obj_id = 0
#
#     def clear(self):
#         self.max_obj_id = 0
#
#     def update(self, track_instances: Instances):
#         # print("track_instances.scores: {}".format(track_instances.scores))
#         track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
#         for i in range(len(track_instances)):
#             if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
#                 # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
#                 track_instances.obj_idxes[i] = self.max_obj_id
#                 self.max_obj_id += 1
#             elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
#                 track_instances.disappear_time[i] += 1
#                 if track_instances.disappear_time[i] >= self.miss_tolerance:
#                     # Set the obj_id to -1.
#                     # Then this track will be removed by TrackEmbeddingLayer.
#                     track_instances.obj_idxes[i] = -1


class PostProcessPanopticInstances(nn.Module):
    """This class converts the output of the model to the final panoptic result, in the format expected by the
    coco panoptic API """

    def __init__(self, is_thing_map={i: i > 10 for i in range(21)}, threshold=0.85,
                 output_dir="", debug=False,
                 fraction_threshold=0.03, pixel_threshold=0.4, apply_mask_removal=False,
                 apply_mask_removal_only_ins=False, use_mask_low_constant=False,
                 catgories_color=None, filter_small_option='4', num_classes=20, num_stuff=11):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the values a boolean indicating whether
                          the class is  a thing (True) or a stuff (False) class
           threshold: confidence threshold: segments with confidence lower than this will be deleted
        """
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

        # for debugging
        self.save_dir = output_dir
        self.debug = debug

        self.catgories_color = catgories_color

        # for mask removal
        self.fraction_threshold = fraction_threshold
        self.pixel_threshold = pixel_threshold
        self.apply_mask_removal = apply_mask_removal
        self.apply_mask_removal_only_ins = apply_mask_removal_only_ins
        self.use_mask_low_constant = use_mask_low_constant

        self.filter_small_option=filter_small_option
        self.num_classes = num_classes
        self.num_stuff = num_stuff

    def mask_removal(self, cls_prob, mask_prob, cls_idx, im_shape, image_id):
        # cur_scores, cur_masks, cur_classes
        # only apply on instances.
        # #1. convert to logits. e.i binary mask. need to set threshould
        # #2. sort by scores.
        # #3. compare the overlap region, if overlap > 0.3 (need adjust), remove
        # #4. if keep, only keep the un-overlapped region. e.i. need to change the mask value.
        mask_prob_copy = mask_prob  # for assign the mask value
        mask_prob = mask_prob.softmax(0)
        cls_prob = cls_prob.detach().cpu().numpy()
        mask_prob = mask_prob.detach().cpu().numpy()
        cls_idx = cls_idx.detach().cpu().numpy()
        mask_prob_copy = mask_prob_copy.detach().cpu().numpy()

        mask_image = np.zeros((np.max(cls_idx)+1,) + im_shape, dtype=mask_prob.dtype)
        panoptic_image = np.zeros(im_shape, dtype=mask_prob.dtype)

        sorted_inds = np.argsort(cls_prob)[::-1]
        cls_prob = cls_prob[sorted_inds]
        cls_idx = cls_idx[sorted_inds]
        mask_prob = mask_prob[sorted_inds]
        mask_prob_copy = mask_prob_copy[sorted_inds]

        # overlap_count = 0
        keep_inds = []
        keep_cls_prob, keep_cls_idx, keep_mask_prob = [], [], []

        if self.apply_mask_removal_only_ins:
            stuff_inds = []
            for i in range(sorted_inds.shape[0]):
                if cls_idx[i] <= self.num_stuff - 1:  #10:
                    stuff_inds.append(i)
                    keep_cls_prob.append(cls_prob[i])
                    keep_cls_idx.append(cls_idx[i])
                    keep_mask_prob.append(mask_prob_copy[i])
                    keep_inds.append(sorted_inds[i])

        for i in range(sorted_inds.shape[0]):
            if self.apply_mask_removal_only_ins and i in stuff_inds:
                continue
            # only handle the instances, so
            # generate logits.
            logit = mask_prob[i].squeeze()
            logit[np.where(logit >= self.pixel_threshold)] = 1
            logit[np.where(logit < self.pixel_threshold)] = 0

            mask_sum = logit.sum()

            curr_mask_image = mask_image[cls_idx[i]]
            if self.debug:
                print("logit_shape: {}, mask_sum: {}, logit_max: {}, logit_min: {}, cls_idx: {},"
                      "overlap_ratio: {}"
                      .format(logit.shape, mask_sum, logit.max(), logit.min(), cls_idx[i],
                              np.logical_and(curr_mask_image >= 1,
                                             logit == 1).sum() / mask_sum
                              ))
            if logit.max() == logit.min() or mask_sum == 0 \
                    or (np.logical_and(curr_mask_image >= 1,
                                       logit == 1).sum() / mask_sum > self.fraction_threshold):
                continue
            assign_mask = np.where(np.logical_and(panoptic_image == 0,
                                       logit == 1))
            keep_cls_prob.append(cls_prob[i])
            keep_cls_idx.append(cls_idx[i])
            if self.use_mask_low_constant:
                empty_mask_prob = np.empty(logit.shape, dtype=logit.dtype)
                empty_mask_prob.fill(_SOFTMAX_MASKING_CONSTANT)
            else:
                empty_mask_prob = np.zeros(logit.shape, dtype=logit.dtype)
            empty_mask_prob[assign_mask] = mask_prob_copy[i][assign_mask]
            keep_mask_prob.append(empty_mask_prob)
            panoptic_image[assign_mask] = 1
            empty_mask_logit = np.zeros(logit.shape, dtype=logit.dtype)
            empty_mask_logit[assign_mask] = logit[assign_mask]
            mask_image[cls_idx[i]] += empty_mask_logit
            keep_inds.append(sorted_inds[i])

            if self.debug:
                print("panoptic_image sum: {}".format(panoptic_image.sum()))
                print("mask_image sum: {}".format(mask_image[cls_idx[i]].sum()))

        if self.debug:
            print(len(sorted_inds), len(keep_cls_prob))

            if len(keep_cls_prob) < len(sorted_inds):
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!, {}, {}".format(len(keep_cls_prob),
                                                                                       len(sorted_inds)))

        keep_cls_prob = np.stack(keep_cls_prob, axis=0)
        keep_cls_idx = np.stack(keep_cls_idx, axis=0)
        keep_mask_prob = np.stack(keep_mask_prob, axis=0)

        return torch.from_numpy(keep_cls_prob).cuda(), torch.from_numpy(keep_mask_prob).cuda(), \
               torch.from_numpy(keep_cls_idx).cuda(), keep_inds

    def forward(self, outputs, processed_sizes, target_sizes=None, id=None):
        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is the detected Instances.
            processed_sizes: This is a list of tuples (or torch tensors) of sizes of the images that were passed to the
                             model, ie the size after data augmentation but before batching.
            target_sizes: This is a list of tuples (or torch tensors) corresponding to the requested final size
                          of each prediction. If left to None, it will default to the processed_sizes
            """
        if target_sizes is None:
            target_sizes = processed_sizes   # [(1024, 2048)]
        assert len(processed_sizes) == len(target_sizes)
        out_logits, raw_masks = outputs.pred_logits.unsqueeze(0), outputs.pred_masks.unsqueeze(0)
        # [1, 100], [1, 100, H, W]
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, size, target_size in zip(
            out_logits, raw_masks, processed_sizes, target_sizes
        ):
            # we filter empty queries and detection below threshold
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            # (100), (100)
            assert self.num_classes in [20, 24, 47]
            if cur_logits.shape[-1] == self.num_classes - 1:  # 19:
                keep = cur_scores > self.threshold
            else:
                keep = cur_classes.ne(cur_logits.shape[-1] - 1) & (cur_scores > self.threshold)
            if self.debug:
                print('keep: {}'.format(keep))
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            if cur_masks.size()[1:] != size:
                cur_masks = interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
            # remember to filter the Instances also.
            outputs = outputs[keep]
            outputs.masks = cur_masks
            outputs.probs = cur_scores
            outputs.labels = cur_classes

            if self.apply_mask_removal:
                cur_scores, cur_masks, cur_classes, keep_inds = self.mask_removal(cur_scores, cur_masks,
                                                                                  cur_classes,
                                                                                  size, id)
                outputs = outputs[keep_inds]
                outputs.masks = cur_masks
                outputs.probs = cur_scores
                outputs.labels = cur_classes

            h, w = cur_masks.shape[-2:]

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class (they are merged later on)
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, classes, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)

                np_seg_img = (
                    torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
                )
                m_id = torch.from_numpy(rgb2id(np_seg_img))

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, cur_classes, dedup=True)
            if cur_classes.numel() > 0:
                loop_times = 0
                # We know filter empty masks as long as we find some
                while True:
                    if self.filter_small_option == '4096_256':
                        filtered_small = torch.as_tensor(
                            [area[i] < 4096 if not self.is_thing_map[c.item()] else area[i] < 256 for i, c in
                             enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                        )
                    elif self.filter_small_option == '4_256':
                        filtered_small = torch.as_tensor(
                            [area[i] < 256 if self.is_thing_map[c.item()] else area[i] < 4 for i, c in
                             enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                        )
                    elif self.filter_small_option == '4':
                        filtered_small = torch.as_tensor(
                            [area[i] <= 4 for i, c in
                             enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                        )
                    else:
                        assert 1 == 2, "filter_small_option is not valid !!!!!!"
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        # remember to filter the Instances also.
                        outputs = outputs[~filtered_small]

                        area, seg_img = get_ids_area(cur_masks, cur_scores, cur_classes)
                    else:
                        break
                    loop_times += 1

            else:
                cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)

            if self.debug:
                print('area: {}'.format(area))
                print("loop_times: {}".format(loop_times))
            segments_info = []
            for i, a in enumerate(area):
                cat = cur_classes[i].item()
                segments_info.append({"id": i, "isthing": self.is_thing_map[cat], "category_id": cat, "area": a})
            del cur_classes

            predictions = {"file_name": 'seg_img_{}.png'.format(id), "segments_info": segments_info}
            preds.append(predictions)
        assert len(preds) == 1
        return outputs