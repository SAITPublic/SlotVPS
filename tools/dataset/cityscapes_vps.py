from __future__ import print_function

import cv2
import os
import os.path as osp
import sys
import torch
import torch.multiprocessing as multiprocessing
import pickle
import numpy as np
import json
import time
from PIL import Image
from tools.config.config import config
from tools.dataset.base_dataset import BaseDataset
from tools.utils import save_color_map
import pdb

def save_instances(segs, pans, cls_inds, obj_ids, names, start_frame, num): #start_frame, num for the sequence to tested.
  cat = 5 #bus, cls_inds are just for thing, start from 1 (person=1)
  f = open('instances.txt','w')
  f.write('hello man!')
  for i in range(start_frame, start_frame+num):
   seg, pan, cls_ind, obj_id, name = segs[i], pans[i], cls_inds[i], obj_ids[i], names[i]
   if len(obj_id) != len(cls_ind):
      f.write('length not correct!\n')
   length = min(len(obj_id), len(cls_ind))
   for j in range(length):
     if cls_ind[j] == cat:
       f.write('frame %d, %d-th ins, obj_id %d, type %d : %s\n' % (i, j, obj_id[j], cls_ind[j], name)) 
  f.close()
   

class CityscapesVps(BaseDataset):

    def __init__(self):

        super(CityscapesVps, self).__init__()

        self.nframes_per_video = 6
        self.lambda_ = 5
        self.labeled_fid = 20

    def inference_panoptic_video(self, pred_pans_2ch, output_dir,
                                 # pan_im_json_file,
                                 categories,
                                 names,
                                 n_video=0):
        from panopticapi.utils import IdGenerator #, IdGenerator_fixed

        # Sample only frames with GT annotations.
        if len(pred_pans_2ch) == 1500:
            pred_pans_2ch = pred_pans_2ch[(self.labeled_fid // self.lambda_)::self.lambda_]
        print('pred_pans_2ch length: {}'.format(len(pred_pans_2ch)))
        categories = {el['id']: el for el in categories}
        color_generator = IdGenerator(categories) #IdGenerator_fixed(categories)   #the new generator (fixed version) is used to be used in multi-threading env.

        def get_pred_large(pan_2ch_all, vid_num, nframes_per_video=6):
            vid_num = len(pan_2ch_all) // nframes_per_video  # 10
            cpu_num = multiprocessing.cpu_count() // 2  # 32 --> 16
            nprocs = min(vid_num, cpu_num)  # 10
            max_nframes = cpu_num * nframes_per_video
            nsplits = (len(pan_2ch_all) - 1) // max_nframes + 1
            annotations, pan_all = [], []
            #pdb.set_trace()
            for i in range(0, len(pan_2ch_all), max_nframes):
                print('==> Read and convert VPS output - split %d/%d' % ((i // max_nframes) + 1, nsplits))
                pan_2ch_part = pan_2ch_all[i:min(
                    i + max_nframes, len(pan_2ch_all))]
                pan_2ch_split = np.array_split(pan_2ch_part, nprocs)
                workers = multiprocessing.Pool(processes=nprocs)
                processes = []
                for proc_id, pan_2ch_set in enumerate(pan_2ch_split): #nprocs * nframes_per_video, in fact, each worker processes one video.
                    p = workers.apply_async(
                        self.converter_2ch_track_core,
                        (proc_id, pan_2ch_set, color_generator))
                    processes.append(p)
                workers.close()
                workers.join()

                for p in processes:
                    p = p.get()
                    annotations.extend(p[0])
                    pan_all.extend(p[1])

            pan_json = {'annotations': annotations}
            return pan_all, pan_json
            
        def get_pred_large_single_thread(pan_2ch_all, vid_num, nframes_per_video=6):
            vid_num = len(pan_2ch_all) // nframes_per_video  # 10
            cpu_num = multiprocessing.cpu_count() // 2  # 32 --> 16
            nprocs = min(vid_num, cpu_num)  # 10
            max_nframes = cpu_num * nframes_per_video
            nsplits = (len(pan_2ch_all) - 1) // max_nframes + 1
            annotations, pan_all = [], []
            for i in range(0, len(pan_2ch_all), max_nframes):
                #pdb.set_trace()  
                print('==> Read and convert VPS output - split %d/%d' % ((i // max_nframes) + 1, nsplits))
                pan_2ch_part = pan_2ch_all[i:min(
                    i + max_nframes, len(pan_2ch_all))]
                pan_2ch_split = np.array_split(pan_2ch_part, nprocs)
                #workers = multiprocessing.Pool(processes=nprocs)
                #processes = []
                for proc_id, pan_2ch_set in enumerate(pan_2ch_split):
                    anno, pan = self.converter_2ch_track_core(proc_id, pan_2ch_set, color_generator)
                    annotations.extend(anno)
                    pan_all.extend(pan)

            pan_json = {'annotations': annotations}
            return pan_all, pan_json

        def save_image(images, save_folder, names, colors=None):
            os.makedirs(save_folder, exist_ok=True)

            names = [osp.join(save_folder,
                              name.replace('_leftImg8bit', '').replace('_newImg8bit', '').replace('jpg', 'png').replace(
                                  'jpeg', 'png')) for name in names]
            cpu_num = multiprocessing.cpu_count() // 2
            images_split = np.array_split(images, cpu_num)
            names_split = np.array_split(names, cpu_num)
            workers = multiprocessing.Pool(processes=cpu_num)
            for proc_id, (images_set, names_set) in enumerate(zip(images_split, names_split)):
                workers.apply_async(BaseDataset._save_image_single_core, (proc_id, images_set, names_set, colors))
            workers.close()
            workers.join()

        # inference_panoptic_video
        pred_pans, pred_json = get_pred_large_single_thread(pred_pans_2ch, vid_num=n_video)
        print('--------------------------------------')
        print('==> Saving VPS output png files')
        os.makedirs(output_dir, exist_ok=True)
        save_image(pred_pans_2ch, osp.join(output_dir, 'pan_2ch'), names)
        save_image(pred_pans, osp.join(output_dir, 'pan_pred'), names)
        print('==> Saving pred.jsons file')
        json.dump(pred_json, open(osp.join(output_dir, 'pred.json'), 'w'))
        print('--------------------------------------')

        return pred_pans, pred_json

    def converter_2ch_track_core(self, proc_id, pan_2ch_set, color_generator):
        from panopticapi.utils import rgb2id

        OFFSET = 1000
        VOID = 255
        annotations, pan_all = [], []
        # reference dict to used color
        inst2color = {}
        seq_ids = [0] * 20  #19 is enough.
        #pdb.set_trace()
        for idx in range(len(pan_2ch_set)): #pan_2ch_set has 6 frames, for an video. 
        #[1024, 2048, 3], the three elements are semantic label (0 to 18), ins_id (1,2,3,..), obj_id (start from 1, both for stuff and thing) 
            pan_2ch = np.uint32(pan_2ch_set[idx])
            pan = OFFSET * pan_2ch[:, :, 0] + pan_2ch[:, :, 2]  #0, pan_seg; 1. pan_ins; 2, pan_obj. 
            pan_format = np.zeros((pan_2ch.shape[0], pan_2ch.shape[1], 3), dtype=np.uint8)
            l = np.unique(pan)
            # segm_info = []
            segm_info = {}
            for el in l:
                sem = el // OFFSET  #semantic label.
                obj_idx = el % OFFSET 
                #print("original el: {}, sem: {}".format(el, sem))
                if sem == VOID or obj_idx==VOID: #zhh: obj_idx is 255, for unknown_mask.
                    continue
                mask = pan == el
                #### handling used color for inst id
                if el % OFFSET > 0:
                    if sem >= 21:
                        sem -= 10
                    # if el > OFFSET:
                    # things class
                    #print("el: {}, sem: {}, seq_ids: {}".format(el, sem, seq_ids))
                    if el in inst2color:
                        color = inst2color[el]
                    else:
                        color = color_generator.get_color(sem, seq_ids[sem])
                        seq_ids[sem] += 1
                        inst2color[el] = color
                else:
                    #print("sem: {}".format(sem))
                    # stuff class
                    color = color_generator.get_color(sem, -1)

                pan_format[mask] = color
                index = np.where(mask)
                x = index[1].min()
                y = index[0].min()
                width = index[1].max() - x
                height = index[0].max() - y

                dt = {"category_id": sem.item(), "iscrowd": 0, "id": int(rgb2id(color)),
                      "bbox": [x.item(), y.item(), width.item(), height.item()], "area": mask.sum().item()}
                segment_id = int(rgb2id(color))
                segm_info[segment_id] = dt

            # annotations.append({"segments_info": segm_info})
            pan_all.append(pan_format)

            gt_pan = np.uint32(pan_format)
            pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
            labels, labels_cnt = np.unique(pan_gt, return_counts=True)
            for label, area in zip(labels, labels_cnt):
                if label == 0:
                    continue
                if label not in segm_info.keys():
                    print('label:', label)
                    raise KeyError('label not in segm_info keys.')

                segm_info[label]["area"] = int(area)
            segm_info = [v for k, v in segm_info.items()]

            annotations.append({"segments_info": segm_info})

        return annotations, pan_all
#solve the contradiction between seg and instance results (segs vs. pans).
    def get_unified_pan_result(self, segs, pans, cls_inds, obj_ids=None, stuff_area_limit=4 * 64 * 64, names=None):
        if obj_ids is None:
            obj_ids = [None for _ in range(len(cls_inds))]
        pred_pans_2ch = {}
        figs = []
        max_oid = 100
        #pdb.set_trace()
        # save_instances(segs, pans, cls_inds, obj_ids, names, 270*5, 30)
        #pdb.set_trace()
        print('segs: {}, {}'.format(segs, len(segs)))
        print('pans: {}, {}'.format(pans, len(pans)))
        print('cls_inds: {}, {}'.format(cls_inds, len(cls_inds)))
        print('obj_ids: {}, {}'.format(obj_ids, len(obj_ids)))
        print('names: {}, {}'.format(names, len(names)))

        temp_count = 0
        for (seg, pan, cls_ind, obj_id, name) in zip(segs, pans, cls_inds, obj_ids, names):
            # handle redundant obj_ids
            if obj_id is not None:
                oid_unique, oid_cnt = np.unique(obj_id, return_counts=True)
                obj_id_ = obj_id[::-1].copy()
                if np.any(oid_cnt > 1):
                    redundants = oid_unique[oid_cnt > 1]
                    for red in redundants:
                        part = obj_id[obj_id == red]
                        for i in range(1, len(part)):
                            part[i] = max_oid
                            max_oid += 1
                        obj_id_[obj_id_ == red] = part
                    obj_id = obj_id_[::-1]

            #print("cls_ind: {}, {}".format(cls_ind, cls_ind.shape))
            #print("obj_id: {}, {}".format(obj_id, obj_id.shape))

            pan_seg = pan.copy()  #output, map for semantic label.
            id_last_stuff = config.dataset.num_seg_classes - config.dataset.num_classes  # =10 = 19 - 9
            if len(cls_ind)==0: #zhanghui: bug fixed now, but initial data has such problem.
              pan[pan>id_last_stuff] = 255
            pan_ins = pan.copy()  #output, map for instance id (start from 1).
            pan_obj = pan.copy()  #output, map for obj id (obj_id+1)
            ids = np.unique(pan)
            ids_ins = ids[ids > id_last_stuff]
            #pan_obj[pan_ins <= id_last_stuff] = 0  #zhanghui added: to clear stuff pixels.
            pan_ins[pan_ins <= id_last_stuff] = 0  #on instance picture, set stuff pixels to 0.
            # print("ids_ins: {}".format(ids_ins))
            for idx, id in enumerate(ids_ins):
                region = (pan_ins == id)
                if id == 255:
                    pan_seg[region] = 255
                    pan_ins[region] = 0
                    continue
                cls, cnt = np.unique(seg[region], return_counts=True)
                # print("cls: {}, cnt: {}, np.argmax(cnt): {}".format(cls, cnt, np.argmax(cnt)))
                if cls[np.argmax(cnt)] == cls_ind[id - id_last_stuff - 1] + id_last_stuff:
                    pan_seg[region] = cls_ind[id - id_last_stuff - 1] + id_last_stuff
                    pan_ins[region] = idx + 1 #start from 1. values are 1,2,3,4... no skip.
                    if obj_id is not None:
                        pan_obj[region] = obj_id[idx] + 1 #obj_id value + 1.
                else:
                    if np.max(cnt) / np.sum(cnt) >= 0.5 and cls[np.argmax(cnt)] <= id_last_stuff:
                        pan_seg[region] = cls[np.argmax(cnt)]
                        pan_ins[region] = 0
                        pan_obj[region] = 0
                    else:
                        pan_seg[region] = cls_ind[id - id_last_stuff - 1] + id_last_stuff
                        pan_ins[region] = idx + 1  
                        if obj_id is not None:
                            pan_obj[region] = obj_id[idx] + 1

            idx_sem = np.unique(pan_seg)
            #print("idx_sem: {}".format(idx_sem))
            for i in range(idx_sem.shape[0]):
                if idx_sem[i] <= id_last_stuff:
                    area = pan_seg == idx_sem[i]
                    if (area).sum() < stuff_area_limit:
                        pan_seg[area] = 255

            pan_2ch = np.zeros((pan.shape[0], pan.shape[1], 3), dtype=np.uint8)
            pan_2ch[:, :, 0] = pan_seg
            pan_2ch[:, :, 1] = pan_ins
            pan_2ch[:, :, 2] = pan_obj

            pred_pans_2ch[name] = pan_2ch

            # cv2.imwrite("pan_2ch_{}.png".format(temp_count), pan_2ch)
            # # save_color_map(pan_2ch, "pan_2ch_{}.png".format(temp_count), apply_color_map=False, clip=False)
            #
            # temp_count += 1
        return pred_pans_2ch
