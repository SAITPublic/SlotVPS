import os.path as osp
import warnings

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from ..registry import PIPELINES
import pdb

@PIPELINES.register_module
class LoadImageFromFile(object):

    def __init__(self, to_float32=False, with_pad_mask=False):
        self.to_float32 = to_float32
        self.with_pad_mask = with_pad_mask

    def __call__(self, results):
        filename = osp.join(results['img_prefix'],
                            results['img_info']['filename'])
        img = mmcv.imread(filename)
        if self.to_float32:
            img = img.astype(np.float32)

        if self.with_pad_mask:
            pad_mask = np.zeros((img.shape[0], img.shape[1]), dtype=int)

        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['iid'] = results['img_info']['id']

        if self.with_pad_mask:
            results['pad_mask'] = pad_mask

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)


@PIPELINES.register_module
class LoadRefImageFromFile(object):
    """
    Code reading reference frame information.
    Specific to Cityscapes-VPS, Cityscapes, and VIPER datasets.
    """
    def __init__(self, sample=True, to_float32=False, with_pad_mask=False):
        self.to_float32 = to_float32
        self.sample = sample
        self.with_pad_mask = with_pad_mask

    def __call__(self, results):
        # requires dirname for ref images
        assert results['ref_prefix'] is not None, 'ref_prefix must be specified.'
        
        filename = osp.join(results['img_prefix'],
                            results['img_info']['filename'])
        img = mmcv.imread(filename)
        if self.to_float32:
            img = img.astype(np.float32)

        if self.with_pad_mask:
            pad_mask = np.zeros((img.shape[0], img.shape[1]), dtype=int)

        # if specified by another ref json file.
        if 'ref_filename' in results['img_info']:
            # added for multiple reference frames
            if isinstance(results['img_info']['ref_filename'], list):
                # TODO not implement with with_pad_mask
                ref_img = []
                ref_pad_mask = []
                for file_name in results['img_info']['ref_filename']:
                    ref_filename = osp.join(results['ref_prefix'], file_name)
                    per_ref_img = mmcv.imread(ref_filename) # [1024, 2048, 3]
                    if self.to_float32:
                        per_ref_img = per_ref_img.astype(np.float32)
                    ref_img.append(per_ref_img)

                    if self.with_pad_mask:
                        ref_pad_mask.append(np.zeros((per_ref_img.shape[0], per_ref_img.shape[1]), dtype=int))
            else:
                ref_filename = osp.join(results['ref_prefix'],
                                        results['img_info']['ref_filename'])
                ref_img = mmcv.imread(ref_filename) # [1024, 2048, 3]
                if self.to_float32:
                    ref_img = ref_img.astype(np.float32)

                if self.with_pad_mask:
                    ref_pad_mask = np.zeros((ref_img.shape[0], ref_img.shape[1]), dtype=int)
        else:
            raise NotImplementedError('We need this implementation.')

        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['ref_img'] = ref_img  # may be a list or single image
        results['iid'] = results['img_info']['id']
        # new added for adding supervision for reference imgs.
        # print("results['img_info'].keys(): {}".format(results['img_info'].keys()))
        # results['img_info'].keys(): dict_keys(['id', 'file_name', 'mode', 'width', 'height', 'date_captured',
        # 'license', 'coco_url', 'flickr_url', 'filename', 'ref_id', 'ref_filename'])

        if self.with_pad_mask:
            results['pad_mask'] = pad_mask
            results['ref_pad_mask'] = ref_pad_mask

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)


@PIPELINES.register_module
class LoadAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 skip_img_without_anno=True,
                 semantic2label=None,
                 with_pid=False,
                 # with_flow=False
                 ):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        # self.with_flow = with_flow
        self.with_pid = with_pid
        self.poly2mask = poly2mask
        self.skip_img_without_anno = skip_img_without_anno
        self.semantic2label = semantic2label

    def _load_bboxes(self, results):

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']
        if len(results['gt_bboxes']) == 0 and self.skip_img_without_anno:
            file_path = osp.join(results['img_prefix'],
                                 results['img_info']['filename'])
            warnings.warn(
                'Skip the image "{}" that has no valid gt bbox'.format(
                    file_path))
            return None
        results['gt_bboxes_ignore'] = ann_info.get('bboxes_ignore', None)
        results['bbox_fields'].extend(['gt_bboxes', 'gt_bboxes_ignore'])
        # if reference annotation,
        if 'ref_ann_info' in results:
            # added for multiple reference frames
            if isinstance(results['ref_ann_info'], list):
                results['ref_bboxes'], results['ref_bboxes_ignore'] = [], []
                for i in range(len(results['ref_ann_info'])):
                    ref_ann_info = results['ref_ann_info'][i]
                    results['ref_bboxes'].append(ref_ann_info['bboxes'])
                    if len(results['ref_bboxes']) == 0 and self.skip_img_without_anno:
                        file_path = osp.join(results['ref_prefix'],
                                             results['img_info']['ref_filename'][i])
                        warnings.warn(
                            'Skip the image "{}" that has no valid gt bbox'.format(file_path))
                        return None
                    results['ref_bboxes_ignore'].append(ref_ann_info.get(
                        'bboxes_ignore', None))
                    # results['ref_bbox_fields'][i].extend(
                    #     ['ref_bboxes', 'ref_bboxes_ignore'])
                    # results['ref_ann_info'][i] = ref_ann_info
                results['ref_bbox_fields'].extend(
                    ['ref_bboxes', 'ref_bboxes_ignore'])
            else:
                ref_ann_info = results['ref_ann_info']
                results['ref_bboxes'] = ref_ann_info['bboxes']
                if len(results['ref_bboxes']) == 0 and self.skip_img_without_anno:
                    file_path = osp.join(results['ref_prefix'],
                                         results['img_info']['ref_filename'])
                    warnings.warn(
                        'Skip the image "{}" that has no valid gt bbox'.format(file_path))
                    return None
                results['ref_bboxes_ignore'] = ref_ann_info.get(
                        'bboxes_ignore', None)
                results['ref_bbox_fields'].extend(
                        ['ref_bboxes', 'ref_bboxes_ignore'])

        return results


    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        #### create obj ids
        if 'obj_ids' in results['ann_info']:
            results['gt_obj_ids'] = results['ann_info']['obj_ids']
        else:
            results['gt_obj_ids'] = np.array(
                    [_ for _ in range(len(results['gt_labels']))])
        # if reference annotations
        if 'ref_ann_info' in results:
            if isinstance(results['ref_ann_info'], list):
                results['ref_labels'], results['ref_obj_ids'] = [], []
                for i in range(len(results['ref_ann_info'])):
                    results['ref_labels'].append(results['ref_ann_info'][i]['labels'])
                    #### create obj ids
                    if 'obj_ids' in results['ref_ann_info'][i]:
                        results['ref_obj_ids'].append(results['ref_ann_info'][i]['obj_ids'])
                    else:
                        results['ref_obj_ids'].append(np.array(
                            [_ for _ in range(len(results['gt_labels']))]))
            else:
                results['ref_labels'] = results['ref_ann_info']['labels']
                #### create obj ids
                if 'obj_ids' in results['ref_ann_info']:
                    results['ref_obj_ids'] = results['ref_ann_info']['obj_ids']
                else:
                    results['ref_obj_ids'] = np.array(
                            [_ for _ in range(len(results['gt_labels']))])

        return results


    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)

        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')

        if 'ref_ann_info' in results:
            # added for multiple reference frames
            if isinstance(results['ref_ann_info'], list):
                results['ref_masks'] = []
                for i in range(len(results['ref_ann_info'])):
                    curr_res = results['ref_ann_info'][i]
                    gt_masks = curr_res['masks']
                    if self.poly2mask:
                        gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
                    results['ref_masks'].append(gt_masks)
                    # curr_res['ref_mask_fields'].append('ref_masks')
                    # results['ref_ann_info'][i] = curr_res
                results['ref_mask_fields'].append('ref_masks')
            else:
                gt_masks = results['ref_ann_info']['masks']
                if self.poly2mask:
                    gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
                results['ref_masks'] = gt_masks
                results['ref_mask_fields'].append('ref_masks')

        return results

    def _load_semantic_seg(self, results):
        seg_filename = results['ann_info']['seg_filename']
        print("seg_filename: {}".format(seg_filename))
        try:
            gt_seg = mmcv.imread(seg_filename, flag='unchanged').squeeze()
        except Exception as e:
            print('Caught exception in worker thread: {}, seg_file_name: {}'.format(e, seg_filename))
        # gt_seg = mmcv.imread(seg_filename, flag='unchanged').squeeze()
        gt_seg_ = gt_seg.copy()
        gt_seg_unique = np.unique(gt_seg)
        for i in gt_seg_unique:
            gt_seg[gt_seg_==i] = self.semantic2label[i]
        results['gt_semantic_seg'] = gt_seg

        if 'ref_ann_info' in results:
            # added for multiple reference frames
            if isinstance(results['ref_ann_info'], list):
                results['ref_semantic_seg'] = []
                for i in range(len(results['ref_ann_info'])):
                    curr_res = results['ref_ann_info'][i]
                    seg_filename = curr_res['seg_filename']
                    gt_seg = mmcv.imread(seg_filename, flag='unchanged').squeeze()
                    gt_seg_ = gt_seg.copy()
                    gt_seg_unique = np.unique(gt_seg)
                    for j in gt_seg_unique:
                        gt_seg[gt_seg_ == j] = self.semantic2label[j]
                    results['ref_semantic_seg'].append(gt_seg)
                    # results['ref_ann_info'][i] = curr_res
            else:
                seg_filename = results['ref_ann_info']['seg_filename']
                gt_seg = mmcv.imread(seg_filename, flag='unchanged').squeeze()
                gt_seg_ = gt_seg.copy()
                gt_seg_unique = np.unique(gt_seg)
                for i in gt_seg_unique:
                    gt_seg[gt_seg_==i] = self.semantic2label[i]
                results['ref_semantic_seg'] = gt_seg

        return results


    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, 
                            self.with_label, self.with_mask, 
                            self.with_seg)
        return repr_str


@PIPELINES.register_module
class LoadProposals(object):

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                'but found {}'.format(proposals.shape))
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([0, 0, 0, 0], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(num_max_proposals={})'.format(
            self.num_max_proposals)
