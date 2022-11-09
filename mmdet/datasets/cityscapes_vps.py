import os
import os.path as osp
from .coco import CocoDataset
from pycocotools.coco import COCO
import numpy as np
from .registry import DATASETS
from mmcv.parallel import DataContainer as DC
from .pipelines.formating import to_tensor
from .pipelines import Compose
import pdb
from numpy import random

@DATASETS.register_module
class CityscapesVPSDataset(CocoDataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix=None,
                 seg_prefix=None,
                 # flow_prefix=None,
                 ref_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 ref_ann_file=None,
                 offsets=None,
                 offsets_change_prob=None,
                 nframes_span_test=6,
                 ref_num_test=1):
        super(CityscapesVPSDataset, self).__init__(
                 ann_file=ann_file,
                 pipeline=pipeline,
                 data_root=data_root,
                 img_prefix=img_prefix,
                 seg_prefix=seg_prefix,
                 # flow_prefix=flow_prefix,
                 ref_prefix=ref_prefix,
                 proposal_file=proposal_file,
                 test_mode=test_mode,
                 ref_ann_file=ref_ann_file)
        self.pipeline = Compose(pipeline)

        if self.ref_ann_file is not None:
            assert self.ref_ann_file == self.ann_file, "ref_ann_file: {}, ann_file: {}".format(ref_ann_file, ann_file)
            self.ref_img_infos = self.load_ref_annotations(
                    self.ref_ann_file)
            self.iid2ref_img_infos = {x['id']:x for x in self.img_infos}
        self.offsets = offsets
        self.offsets_change_prob = offsets_change_prob
        self.nframes_span_test = nframes_span_test
        self.ref_num_test = ref_num_test

        self.iid2img_infos = {x['id']: x for x in self.img_infos}
        # to get all previous ids in the same video
        self.vid_dict = {}
        for img_info in self.img_infos:
            self.vid_dict[img_info['id'] // 10000] = self.vid_dict.get(img_info['id'] // 10000, [])
            self.vid_dict[img_info['id'] // 10000].append(img_info['id'])
        # sort iid for each vid
        for k in self.vid_dict.keys():
            self.vid_dict[k] = sorted(self.vid_dict[k])

    CLASSES = ('person', 'rider', 'car', 'truck', 'bus',
               'train', 'motorcycle', 'bicycle')

    def load_ref_annotations(self, ann_file):
        self.ref_coco = COCO(ann_file)
        self.ref_cat_ids = self.ref_coco.getCatIds()
        self.ref_cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.ref_cat_ids)
        }
        self.ref_img_ids = self.ref_coco.getImgIds()
        img_infos = []
        for i in self.ref_img_ids:
            info = self.ref_coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos


    def get_ref_ann_info_by_iid(self, img_id, ref_img_info):
        # img_id = self.ref_img_infos[idx]['id']
        ann_ids = self.ref_coco.getAnnIds(imgIds=[img_id])
        ann_info = self.ref_coco.loadAnns(ann_ids)
        return self._parse_ann_info(ref_img_info, ann_info)


    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['ref_prefix'] = self.ref_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['ref_bbox_fields'] = []
        results['ref_mask_fields'] = []

    def prepare_train_img(self, idx):
        # len(self.img_infos) # 2364
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)

        random_value = random.random()

        iid = img_info['id']
        # self.offsets = [-1, 1] for Cityscapes_VPS
        if self.offsets == '0' or (self.offsets == '0_or_ref1' and random_value < self.offsets_change_prob):
            ref_ann_info = ann_info
            img_info['ref_id'] = img_info['id']
            img_info['ref_filename'] = img_info['filename']
            if self.offsets == '0_or_ref1':
                img_info['apply_image_shift'] = True
        elif '0_shift' in self.offsets:  # e.g. '0_shift_2'
            img_info['ref_id'] = []
            img_info['ref_filename'] = []
            ref_ann_info = []
            shift_num = int(self.offsets.split("_")[-1])
            for _ in range(shift_num):
                img_info['ref_id'].append(img_info['id'])
                img_info['ref_filename'].append(img_info['filename'])
                ref_ann_info.append(ann_info)
            assert isinstance(ref_ann_info, list)
        elif self.offsets == [-1, 1] or isinstance(self.offsets, list) or \
                (self.offsets == '0_or_ref1' and random_value >= self.offsets_change_prob):
            # offsets = [-1, 1]
            if isinstance(self.offsets, list): # random sample one reference image.
                offsets = self.offsets.copy()
            else:
                assert self.offsets == '0_or_ref1'
                offsets = [-1, 1]
            # print("offsets: {}".format(offsets))
            # random sampling of future or past 5-th frame [-1, 1]
            while True:
                m = np.random.choice(offsets)
                if iid+m in self.ref_img_ids:
                    break
                offsets.remove(m)
                # If all offset values fail, return None.
                if len(offsets)==0:
                    return None
            # Reference image: information, annotations
            ref_iid = iid + m

            ref_img_info = self.iid2ref_img_infos[ref_iid]
            ref_ann_info = self.get_ref_ann_info_by_iid(
                ref_iid, ref_img_info)
            img_info['ref_id'] = ref_img_info['id']
            img_info['ref_filename'] = ref_img_info['filename']
            if self.offsets == '0_or_ref1':
                img_info['apply_image_shift'] = False
        else:
            # for multiple reference frames
            all_iids = self.vid_dict[iid // 10000]
            # only contains the reference iids
            if self.offsets == 'all' or self.offsets == 'full_all':
                start_idx = 0
            elif self.offsets == '-2':
                start_idx = max(0, all_iids.index(iid) - 2)
            elif self.offsets == '-3' or self.offsets == '+-3':
                start_idx = max(0, all_iids.index(iid) - 3)
            elif self.offsets == '-4':
                start_idx = max(0, all_iids.index(iid) - 4)
            else:
                assert 1 == 2, "INVALID OFFSETS !!!!!! {}".format(self.offsets)
            used_iids = all_iids[start_idx:all_iids.index(iid)]
            if self.offsets == 'full_all':
                used_iids += all_iids[all_iids.index(iid)+1:]
            if '+-' in self.offsets:
                ref_iids_len = int(self.offsets[-1])
                used_iids += all_iids[all_iids.index(iid)+1:all_iids.index(iid)+1+(ref_iids_len-len(used_iids))]
                assert len(used_iids) == ref_iids_len, '{}, {}'.format(ref_iids_len, len(used_iids))
            used_idxs = [self.img_infos.index(self.iid2img_infos[u]) for u in used_iids]
            if len(used_idxs) == 0:
                return None

            if len(used_idxs) > 1:
                img_info['ref_id'] = []
                img_info['ref_filename'] = []
                ref_ann_info = []
                for ref_iid in used_iids:
                    ref_img_info = self.iid2ref_img_infos[ref_iid]
                    each_ref_ann_info = self.get_ref_ann_info_by_iid(
                             ref_iid, ref_img_info)
                    img_info['ref_id'].append(ref_img_info['id'])
                    img_info['ref_filename'].append(ref_img_info['filename'])
                    ref_ann_info.append(each_ref_ann_info)
                assert isinstance(ref_ann_info, list)
            else:
                ref_iid = used_iids[0]
                ref_img_info = self.iid2ref_img_infos[ref_iid]
                ref_ann_info = self.get_ref_ann_info_by_iid(
                        ref_iid, ref_img_info)
                img_info['ref_id'] = ref_img_info['id']
                img_info['ref_filename'] = ref_img_info['filename']
                assert not isinstance(img_info['ref_filename'], list)
                assert not isinstance(ref_ann_info, list)

        results = dict(img_info=img_info, ann_info=ann_info,
                       ref_ann_info=ref_ann_info)

        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        ### semantic segmentation label (for target frame)
        # Cityscapes - specific filename
        seg_filename = osp.join(
                results['seg_prefix'],
                results['ann_info']['seg_map'].replace(
                        'leftImg8bit', 'gtFine_color')).replace(
                                'newImg8bit', 'final_mask')
        # print(seg_filename)
        results['ann_info']['seg_filename'] = seg_filename
        ### semantic segmentation label (for reference frame)
        # ===> Not being used in current training implementation.
        if isinstance(ref_ann_info, list):
            for i in range(len(ref_ann_info)):
                results['ref_ann_info'][i]['seg_filename'] = osp.join(results['seg_prefix'],
                                                            results['ref_ann_info'][i]['seg_map'].replace(
                                                                    'leftImg8bit', 'gtFine_color')).replace(
                                                                            'newImg8bit', 'final_mask')
        else:
            ref_seg_filename = osp.join(results['seg_prefix'],
                    results['ref_ann_info']['seg_map'].replace(
                            'leftImg8bit', 'gtFine_color')).replace(
                                    'newImg8bit', 'final_mask')
            results['ref_ann_info']['seg_filename'] = ref_seg_filename

        data = self.pipeline(results)
        if data is None:
            return None
        ### tracking label
        if isinstance(ref_ann_info, list):
            assert len(data['ref_obj_ids']) == len(ref_ann_info)
            # start from the first ref_obj_ids, build the ref_ids memory.
            ref_ids = data['ref_obj_ids'][0].data.numpy().tolist()
            ref_pids = [[ref_ids.index(i) + 1 for i in ref_ids]]
            for each_ref_ids in data['ref_obj_ids'][1:]:
                ref_pid = []
                for id in each_ref_ids.data.numpy().tolist():
                    if id not in ref_ids:
                        ref_ids.append(id)
                        ref_pid.append(ref_ids.index(id) + 1)
                    else:
                        ref_pid.append(ref_ids.index(id) + 1)
                ref_pids.append(ref_pid)
        else:
            ref_ids = data['ref_obj_ids'].data.numpy().tolist()
        gt_ids = data['gt_obj_ids'].data.numpy().tolist()
        gt_pids = [ref_ids.index(i)+1 if i in ref_ids else 0 for i in gt_ids]
        data['gt_pids'] = DC(to_tensor(gt_pids))
        if isinstance(ref_ann_info, list):
            # print("ref_pids: {}".format(ref_pids))
            data['ref_gt_pids'] = ref_pids

        return data

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]

        assert self.ref_num_test == 1
        prev_img_info = self.img_infos[idx - 1] if idx % (self.nframes_span_test) > 0 else img_info
        img_info['ref_id'] = prev_img_info['id'] - 1
        img_info['ref_filename'] = prev_img_info['file_name']

        results = dict(img_info=img_info)

        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.
        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.
        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_obj_ids = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])
                gt_obj_ids.append(ann['inst_id'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_obj_ids = np.array(gt_obj_ids, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_obj_ids = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            obj_ids=gt_obj_ids,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann