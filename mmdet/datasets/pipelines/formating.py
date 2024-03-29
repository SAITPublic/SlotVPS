from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from ..registry import PIPELINES

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


@PIPELINES.register_module
class ToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)


@PIPELINES.register_module
class ImageToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        
        for key in self.keys:
            if key in ['ref_img']:
                if isinstance(results[key], list):
                    img_ref = []
                    for img in results[key]:
                        img = np.ascontiguousarray(img.transpose(2,0,1))
                        img_ref.append(img)
                    img_ref = np.array(img_ref)
                    results[key] = to_tensor(img_ref)
                else:
                    img = np.ascontiguousarray(results[key].transpose(2, 0, 1))
                    results[key] = to_tensor(img)
            else:
                results[key] = to_tensor(results[key].transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)


@PIPELINES.register_module
class Transpose(object):

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={}, order={})'.format(
            self.keys, self.order)


@PIPELINES.register_module
class ToDataContainer(object):

    def __init__(self, fields=(dict(key='img', stack=True), 
                               dict(key='gt_bboxes'),
                               dict(key='gt_labels'))):
        self.fields = fields

    def __call__(self, results):
        for field in self.fields:
            field = field.copy()
            key = field.pop('key')
            results[key] = DC(results[key], **field)
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(fields={})'.format(self.fields)


@PIPELINES.register_module
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    - img_ref: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    """

    def __call__(self, results):

        if 'img' in results:
            img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        if 'ref_img' in results:
            # added for multiple reference frames
            if isinstance(results['ref_img'], list):
                imgs = []
                for each_img in results['ref_img']:
                    img = np.ascontiguousarray(each_img.transpose(2, 0, 1))
                    imgs.append(DC(to_tensor(img), stack=True))
                results['ref_img'] = imgs
            else:
                img = np.ascontiguousarray(results['ref_img'].transpose(2, 0, 1))
                results['ref_img'] = DC(to_tensor(img), stack=True)

        if 'pad_mask' in results:
            results['pad_mask'] = DC(to_tensor(results['pad_mask']))
            results['ref_pad_mask'] = DC(to_tensor(results['ref_pad_mask']))

        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore',
                    'gt_labels', 'gt_obj_ids', 'ref_bboxes', 
                    'ref_bboxes_ignore', 'ref_labels', 'ref_obj_ids',
                    ]:
            if key not in results:
                continue
            if key in ['ref_bboxes', 'ref_bboxes_ignore', 'ref_labels', 'ref_obj_ids'] \
                    and isinstance(results['ref_img'], list):
                for i in range(len(results[key])):
                    results[key][i] = DC(to_tensor(results[key][i]))
            else:
                results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'ref_masks' in results:
            # if 'ref_img' is list, ref_masks must be lists.
            if isinstance(results['ref_img'], list):
                for i in range(len(results['ref_masks'])):
                    results['ref_masks'][i] = DC(results['ref_masks'][i], cpu_only=True)
            else:
                results['ref_masks'] = DC(results['ref_masks'], cpu_only=True)

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None,:,:]), stack=True)
        if 'gt_semantic_seg_Nx' in results:
            results['gt_semantic_seg_Nx'] = DC(
                to_tensor(results['gt_semantic_seg_Nx'][None,:,:]), stack=True)
        if 'ref_semantic_seg' in results:
            if isinstance(results['ref_img'], list):
                for i in range(len(results['ref_semantic_seg'])):
                    results['ref_semantic_seg'][i] = DC(
                        to_tensor(results['ref_semantic_seg'][i][None, :, :]), stack=True)
            else:
                results['ref_semantic_seg'] = DC(
                    to_tensor(results['ref_semantic_seg'][None,:,:]), stack=True)
        if 'ref_semantic_seg_Nx' in results:
            if isinstance(results['ref_img'], list):
                for i in range(len(results['ref_semantic_seg_Nx'])):
                    results['ref_semantic_seg_Nx'][i] = DC(
                        to_tensor(results['ref_semantic_seg_Nx'][i][None, :, :]), stack=True)
            else:
                results['ref_semantic_seg_Nx'] = DC(
                    to_tensor(results['ref_semantic_seg_Nx'][None,:,:]), stack=True)
            
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module
class Collect(object):

    def __init__(self, keys, meta_keys=(
            'filename', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor', 'flip', 'img_norm_cfg','iid')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        # print("results.keys(): {}".format(results.keys()))
        # results.keys(): dict_keys(['img_info', 'ann_info', 'ref_ann_info', 'img_prefix', 'seg_prefix', 'ref_prefix',
        # 'proposal_file', 'bbox_fields', 'mask_fields', 'ref_bbox_fields', 'ref_mask_fields', 'filename', 'img',
        # 'img_shape', 'ori_shape', 'ref_img', 'iid', 'gt_bboxes', 'gt_bboxes_ignore', 'ref_bboxes', 'ref_bboxes_ignore',
        # 'gt_labels', 'gt_obj_ids', 'ref_labels', 'ref_obj_ids', 'gt_masks', 'ref_masks',
        # 'gt_semantic_seg', 'ref_semantic_seg', 'scale', 'scale_idx', 'pad_shape', 'scale_factor',
        # 'keep_ratio', 'flip', 'img_norm_cfg', 'crop_coords', 'pad_fixed_size', 'pad_size_divisor',
        # 'ref_semantic_seg_Nx', 'gt_semantic_seg_Nx'])
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_meta'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            if key not in results:
                data[key] = None
            else:
                data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(keys={}, meta_keys={})'.format(
            self.keys, self.meta_keys)
