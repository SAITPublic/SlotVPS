# -------------------------------------------------------------------
# Video Panoptic Segmentation
#
# VPQ evaluation code by tube (video segment) matching
# Inference on every frames and evaluation on every 5 frames.
# ------------------------------------------------------------------

import argparse
import sys
import os
import os.path
import numpy as np
from PIL import Image
import multiprocessing
import time
import json
from collections import defaultdict
import copy
import pdb
from utils import draw_line_charts, draw_line_chart

class PQStatCat():
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        # added for evaluate the consistency
        self.ids_sum = 0
        self.ids_false = 0
        self.ids_errp = 0

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        # added for evaluate the consistency
        self.ids_sum += pq_stat_cat.ids_sum
        self.ids_false += pq_stat_cat.ids_false
        self.ids_errp += pq_stat_cat.ids_errp
        return self

class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories, isthing):
        # PQ = SQ * RQ. SQ: averaged iou of TP, RQ: recognition quality tp / (tp + 0.5 * fp + 0.5 * fn)
        # average the PQ among all categories.
        # print("isthing: {}".format(isthing))
        pq, sq, rq, n = 0, 0, 0, 0
        ids_sum, ids_false, ids_errp = 0, 0, 0
        per_class_results = {}
        tps, fps, fns = 0, 0, 0
        for label, label_info in categories.items():
            if isthing is not None:
                # double check the isthing flag.
                cat_isthing = label_info['isthing'] == 1
                # print("label_info['isthing']: {}, cat_isthing: {}".format(label_info['isthing'], cat_isthing))
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            # print("iou: {}, tp: {}, fp: {}, fn: {}".format(iou, tp, fp, fn))
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0, 'iou': 0.0, 'tp': 0, 'fp': 0, 'fn': 0,
                                            'ids_sum': 0, 'ids_false': 0, 'ids_errp': 0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, 'iou': iou, 'tp':tp, 'fp':fp, 'fn':fn}
            pq += pq_class
            sq += sq_class
            rq += rq_class

            tps += tp
            fps += fp
            fns += fn

            # added for evaluate the consistency
            per_class_results[label]['ids_sum'] = self.pq_per_cat[label].ids_sum
            per_class_results[label]['ids_false'] = self.pq_per_cat[label].ids_false
            per_class_results[label]['ids_errp'] = self.pq_per_cat[label].ids_false / self.pq_per_cat[label].ids_sum if \
            self.pq_per_cat[label].ids_sum != 0 else 0
            ids_sum += self.pq_per_cat[label].ids_sum
            ids_false += self.pq_per_cat[label].ids_false
            ids_errp += per_class_results[label]['ids_errp']

        # print('pq_average: n: ' + str(n))
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # ids_false equals to the ID SWITCH metric in MOT, therefore not devide by category num.
        if n > 0:
            # default
            return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n, 'ids_sum': ids_sum, 'ids_false': ids_false,
                    'ids_errp': ids_errp, 'tps': tps, 'fps': fps, 'fns': fns}, per_class_results
        else:
            return {'pq': 0, 'sq': 0, 'rq': 0, 'n': 0, 'ids_sum': ids_sum, 'ids_false': ids_false,
                    'ids_errp': ids_errp, 'tps': tps, 'fps': fps, 'fns': fns}, per_class_results


def vpq_compute_single_core(gt_pred_set, categories, nframes=2):
    OFFSET = 256 * 256 * 256
    VOID = 0
    vpq_stat = PQStat()
    # all ids memory bank
    ids_memory = {}

    # Iterate over the video frames 0::T-λ
    # gt_pred_set: gt_jsons, pred_jsons, gt_pans, pred_pans, gt_image_jsons
    # len(gt_pred_set): 6
    for idx in range(0, len(gt_pred_set)-nframes+1): 
        vid_pan_gt, vid_pan_pred = [], []
        gt_segms_list, pred_segms_list = [], []

        # Matching nframes-long tubes.
        # Collect tube IoU, TP, FP, FN
        for i, (gt_json, pred_json, gt_pan, pred_pan, gt_image_json) in enumerate(gt_pred_set[idx:idx+nframes]):
            #### Step1. Collect frame-level pan_gt, pan_pred, etc.
            gt_pan, pred_pan = np.uint32(gt_pan), np.uint32(pred_pan)
            # gt_pan shape: (1024, 2048, 3)
            pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
            pan_pred = pred_pan[:, :, 0] + pred_pan[:, :, 1] * 256 + pred_pan[:, :, 2] * 256 * 256
            # pan_gt shape: (1024, 2048)
            gt_segms = {}
            for el in gt_json['segments_info']:
                if el['id'] in gt_segms:
                    gt_segms[el['id']]['area'] += el['area']
                else:
                    gt_segms[el['id']] = copy.deepcopy(el)   # el
            pred_segms = {}
            for el in pred_json['segments_info']:
                if el['id'] in pred_segms:
                    pred_segms[el['id']]['area'] += el['area']
                else:
                    pred_segms[el['id']] = copy.deepcopy(el)   # el
            # pred_segms: eg:'category_id': 18, 'iscrowd': 0, 'id': 2100087, 'bbox': [666, 427, 12, 54], 'area': 504
            # predicted segments area calculation + prediction sanity checks
            # mainly sanity check, pred_segms not changed
            pred_labels_set = set(el['id'] for el in pred_json['segments_info'])
            labels, labels_cnt = np.unique(pan_pred, return_counts=True)
            for label, label_cnt in zip(labels, labels_cnt):
                if label not in pred_segms:
                    if label == VOID:
                        continue
                    raise KeyError('Segment with ID {} is presented in PNG and not presented in JSON.'.format(label))
                pred_segms[label]['area'] = label_cnt
                pred_labels_set.remove(label)
                if pred_segms[label]['category_id'] not in categories:
                    raise KeyError('Segment with ID {} has unknown category_id {}.'.format(label, pred_segms[label]['category_id']))
            if len(pred_labels_set) != 0:
                raise KeyError(
                    'The following segment IDs {} are presented in JSON and not presented in PNG.'.format(list(pred_labels_set)))

            vid_pan_gt.append(pan_gt)
            vid_pan_pred.append(pan_pred)
            gt_segms_list.append(gt_segms)
            pred_segms_list.append(pred_segms)

        #### Step 2. Concatenate the collected items -> tube-level. 
        vid_pan_gt = np.stack(vid_pan_gt) # [nf,H,W]
        vid_pan_pred = np.stack(vid_pan_pred) # [nf,H,W]
        vid_gt_segms, vid_pred_segms = {}, {}
        for gt_segms, pred_segms in zip(gt_segms_list, pred_segms_list):
            # aggregate into tube 'area'
            for k in gt_segms.keys():
                if not k in vid_gt_segms:
                    vid_gt_segms[k] = copy.deepcopy(gt_segms[k])  # gt_segms[k]
                else:
                    vid_gt_segms[k]['area'] += gt_segms[k]['area']
            for k in pred_segms.keys():
                if not k in vid_pred_segms:
                    vid_pred_segms[k] = copy.deepcopy(pred_segms[k])  # pred_segms[k]
                else:
                    vid_pred_segms[k]['area'] += pred_segms[k]['area']

        #### Step3. Confusion matrix calculation
        vid_pan_gt_pred = vid_pan_gt.astype(np.uint64) * OFFSET + vid_pan_pred.astype(np.uint64)
        # print(vid_pan_gt_pred)
        # print(vid_pan_gt_pred.shape)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(vid_pan_gt_pred, return_counts=True)
        # print('labels', labels)
        # print('labels_cnt', labels_cnt)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection
            # print((gt_id, pred_id))
        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        tp = 0
        fp = 0
        fn = 0

        # print('len(vid_gt_segms): {}, len(vid_pred_segms): {}, len(gt_pred_map): {}'.format(len(vid_gt_segms), len(vid_pred_segms), len(gt_pred_map)))
        #### Step4. Tube matching
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple

            if gt_label not in vid_gt_segms:
                continue
            if pred_label not in vid_pred_segms:
                continue
            if vid_gt_segms[gt_label]['iscrowd'] == 1:
                continue
            # if the predicted category do not match with the gt category, then do not calculate the iou.
            if vid_gt_segms[gt_label]['category_id'] != \
                    vid_pred_segms[pred_label]['category_id']:
                continue

            union = vid_pred_segms[pred_label]['area'] + vid_gt_segms[gt_label]['area'] - intersection - gt_pred_map.get(
                (VOID, pred_label), 0)
            iou = intersection / union
            assert iou <= 1.0, 'INVALID IOU VALUE : %d'%(gt_label)
            # count true positives
            if iou > 0.5:
                vpq_stat[vid_gt_segms[gt_label]['category_id']].tp += 1
                vpq_stat[vid_gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)
                tp += 1

                # for calculating the error percentage.
                vpq_stat[vid_gt_segms[gt_label]['category_id']].ids_sum += 1
                if gt_label not in ids_memory:
                    ids_memory[gt_label] = pred_label
                else:
                    if pred_label != ids_memory[gt_label]:
                        vpq_stat[vid_gt_segms[gt_label]['category_id']].ids_false += 1
                    # elif pred_label != gt_label:
                    #     vpq_stat[vid_gt_segms[gt_label]['category_id']].ids_false += 0.5
                    ids_memory[gt_label] = pred_label

        # count false negatives, not matched GT, in other words, not detected.
        crowd_labels_dict = {}
        # gt_label is instance id.
        for gt_label, gt_info in vid_gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                # print("FN invalid: crowd gt area .....")
                continue
            vpq_stat[gt_info['category_id']].fn += 1
            fn += 1

            # if gt_label not in ids_memory: # means that it is always not detected.
            #     vpq_stat[gt_info['category_id']].ids_false += 0.5
            # else:
            #     vpq_stat[gt_info['category_id']].ids_false += 1
            vpq_stat[gt_info['category_id']].ids_sum += 1

        # count false positives, not matched prediction, have the bounding box, but false prediction.
        for pred_label, pred_info in vid_pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                # print("FP invalid: more than half of the predicted segment correspond to VOID and CROWD regions .....")
                continue
            vpq_stat[pred_info['category_id']].fp += 1
            fp += 1

            # if pred_label not in ids_memory: # means that the prediction is too much out of range
            #     vpq_stat[pred_info['category_id']].ids_false += 1
            # else:
            #     if pred_label != ids_memory[pred_label]:
            #         vpq_stat[pred_info['category_id']].ids_false += 1
            #     else:
            #         vpq_stat[pred_info['category_id']].ids_false += 0.5
            # vpq_stat[pred_info['category_id']].ids_sum += 1

        # print('tp:{}, fp:{}, fn:{}, ids_sum: {}, id_false: {}'.format(tp, fp, fn, ids_sum, id_false))

    return vpq_stat


def vpq_compute(gt_pred_split, categories, nframes, output_dir, all_ys_vpq, all_ys_vsq, all_ys_vrq, all_ys_errp,
                all_cats_vpq, all_cats_vsq, all_cats_vrq, all_cats_errp):
    start_time = time.time()
    vpq_stat = PQStat()
    k = (nframes - 1) * 5
    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    # first calculate the vpq for each category
    # gt_pred_set: gt_jsons, pred_jsons, gt_pans, pred_pans, gt_image_jsons
    for _ in range(3):
        all_ys_vpq.append([])
        all_ys_errp.append([])
        all_ys_vsq.append([])
        all_ys_vrq.append([])
    all_cats_vpq.append([])
    all_cats_vsq.append([])
    all_cats_vrq.append([])
    all_cats_errp.append([])
    for idx, gt_pred_set in enumerate(gt_pred_split):
        tmp = vpq_compute_single_core(gt_pred_set, categories, nframes=nframes)

        tmp_results = {}
        for name, isthing in metrics:
            tmp_results[name], tmp_per_class_results = tmp.pq_average(categories, isthing=isthing)
            if name == 'All':
                tmp_results['per_class'] = tmp_per_class_results
                all_ys_vpq[-3].append(100 * tmp_results[name]['pq'])
                all_ys_vsq[-3].append(100 * tmp_results[name]['sq'])
                all_ys_vrq[-3].append(100 * tmp_results[name]['rq'])
                all_ys_errp[-3].append(100 * tmp_results[name]['ids_errp'])
            elif name == 'Things':
                all_ys_vpq[-2].append(100 * tmp_results[name]['pq'])
                all_ys_vsq[-2].append(100 * tmp_results[name]['sq'])
                all_ys_vrq[-2].append(100 * tmp_results[name]['rq'])
                all_ys_errp[-2].append(100 * tmp_results[name]['ids_errp'])
            elif name == 'Stuff':
                all_ys_vpq[-1].append(100 * tmp_results[name]['pq'])
                all_ys_vsq[-1].append(100 * tmp_results[name]['sq'])
                all_ys_vrq[-1].append(100 * tmp_results[name]['rq'])
                all_ys_errp[-1].append(100 * tmp_results[name]['ids_errp'])

        # print('k, idx:')
        # print((nframes-1)*5, idx)
        # print(100 * tmp_results['All']['pq'], 100 * tmp_results['Things']['pq'],
        #       100 * tmp_results['Stuff']['pq'], 100 * tmp_results['All']['ids_errp'],
        #       100 * tmp_results['Things']['ids_errp'], 100 * tmp_results['Stuff']['ids_errp'])
        vpq_stat += tmp
    # print('***************************************')

    # then pq_average to average the vpqs among all categories.
    # hyperparameter: window size k
    print('==> %d-frame vpq_stat:'%(k), time.time()-start_time, 'sec')
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = vpq_stat.pq_average(categories, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results

    vpq_all = 100 * results['All']['pq']
    vpq_thing = 100 * results['Things']['pq']
    vpq_stuff = 100 * results['Stuff']['pq']
    # ids_errp = 100 * results['All']['ids_errp']
    ids_errp = 100 * (results['All']['ids_false'] / results['All']['ids_sum'])
    thing_errp = 100 * results['Things']['ids_errp']  # actually no use
    stuff_errp = 100 * results['Stuff']['ids_errp']   # actually no use

    vsq_all = 100 * results['All']['sq']
    vrq_all = 100 * results['All']['rq']

    save_name = os.path.join(output_dir, 'vpq-%d.txt'%(k))
    f = open(save_name, 'w') if save_name else None
    f.write("================================================\n")
    f.write("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s} {:>5s} {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N", "ERRP", "SUM", "FALSE\n"))
    f.write("-" * (10 + 7 * 7)+'\n')
    for name, _isthing in metrics:
        f.write("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d} {:5.1f} {:5.1f} {:5.1f}\n"
                .format(name, 100 * results[name]['pq'], 100 * results[name]['sq'],
                        100 * results[name]['rq'], results[name]['n'],
                        100 * results[name]['ids_errp'],
                        results[name]['ids_sum'],
                        results[name]['ids_false']))
    f.write("{:4s}| {:>5s} {:>5s} {:>5s} {:>6s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s}\n"
            .format("IDX", "PQ", "SQ", "RQ", "IoU", "TP", "FP", "FN", "ERRP", "SUM", "FALSE"))
    for idx, result in results['per_class'].items():
        f.write("{:4d} | {:5.1f} {:5.1f} {:5.1f} {:6.1f} {:7d} {:7d} {:7d} {:7.1f} {:7.1f} {:7.1f}\n"
                .format(idx, 100 * result['pq'], 100 * result['sq'],
                        100 * result['rq'], result['iou'], result['tp'], result['fp'],
                        result['fn'], 100 * result['ids_errp'],
                        result['ids_sum'], result['ids_false']))
        all_cats_vpq[-1].append(100 * result['pq'])
        all_cats_vsq[-1].append(100 * result['sq'])
        all_cats_vrq[-1].append(100 * result['rq'])
        all_cats_errp[-1].append(100 * result['ids_errp'])

    if save_name:
        f.close()

    if k == 0:
        print("================================================")
        print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s} {:>5s} {:>5s} {:>5s}  {:>5s}  {:>5s}  {:>5s}".format("", "PQ", "SQ",
                                                                                                          "RQ", "N", "ERRP",
                                                                                                          "SUM", "FALSE",
                                                                                                          "TPS", "FPS",
                                                                                                          "FNS"))
        print("-" * (10 + 7 * 7) + '\n')
        for name, _isthing in metrics:
            print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d} {:5.1f} {:5.1f} {:5.1f} {:5.1f} {:5.1f} {:5.1f}"
                    .format(name, 100 * results[name]['pq'], 100 * results[name]['sq'],
                            100 * results[name]['rq'], results[name]['n'],
                            # 100 * results[name]['ids_errp'],
                            100 * (results[name]['ids_false'] / results[name]['ids_sum']),
                            results[name]['ids_sum'],
                            results[name]['ids_false'],
                            results[name]['tps'],
                            results[name]['fps'],
                            results[name]['fns']))
    return vsq_all, vrq_all, vpq_all, vpq_thing, vpq_stuff, ids_errp, thing_errp, stuff_errp, all_ys_vpq, all_ys_vsq, \
           all_ys_vrq, all_ys_errp, all_cats_vpq, all_cats_vsq, all_cats_vrq, all_cats_errp


def final_eval(args, submit_dir, truth_dir, output_dir,curr_checkpoint=10000):
    start_all = time.time()
    pan_pred_json_file = os.path.join(submit_dir, 'pred.json')
    with open(pan_pred_json_file, 'r') as f:
        pred_jsons = json.load(f)
    pan_gt_json_file = args.pan_gt_json_file
    with open(pan_gt_json_file, 'r') as f:
        gt_jsons = json.load(f)

    # gt_jsons keys: images, annotations, categories
    # pred_jsons keys: annotations
    #zhanghui: only remain the video length with prediction results.
    len_pred = len(pred_jsons['annotations'])
    len_gt = len(gt_jsons['annotations'])
    if len_pred<len_gt:
      gt_jsons['images'] = gt_jsons['images'][:len_pred]
      gt_jsons['annotations'] = gt_jsons['annotations'][:len_pred]

    cats_x = []
    for el in gt_jsons['categories']:
        cats_x.append(el['name'])
    print(cats_x)

    categories = gt_jsons['categories']
    categories = {el['id']: el for el in categories}
    # ==> pred_json, gt_json, categories
    # len(categories): 19.  id: {categories[i]}

    # load gt mask images to gt_pans and prediction mask images to pred_pans.
    start_time = time.time()
    gt_pans = []
    files = [
        item['file_name'].replace('_newImg8bit.png', '_final_mask.png').replace('_leftImg8bit.png', '_gtFine_color.png')
        for item in gt_jsons['images']]
    files.sort()
    for idx, file in enumerate(files):
        image = np.array(Image.open(os.path.join(truth_dir, file)))
        gt_pans.append(image)
    print('==> gt_pans:', len(gt_pans), '//', time.time() - start_time, 'sec')

    start_time = time.time()
    pred_pans = []
    files = [item['id'] + '.png' for item in gt_jsons['images']]
    for idx, file in enumerate(files):
        image = np.array(Image.open(os.path.join(submit_dir, 'pan_pred', file)))
        pred_pans.append(image)
        if args.save_diff_fig:
            # generate the error map here.
            diff_pan = image - gt_pans[idx]
            diff_pan[np.where(gt_pans[idx] == 0)] = 0
            diff_image = Image.fromarray(diff_pan)
            os.makedirs(os.path.join(submit_dir, 'pan_diff'), exist_ok=True)
            diff_image.save(os.path.join(submit_dir, 'pan_diff', file))
    print('==> pred_pans:', len(pred_pans), '//', time.time() - start_time, 'sec')
    assert len(gt_pans) == len(pred_pans), "number of prediction does not match with the groud truth."

    gt_image_jsons = gt_jsons['images']
    gt_jsons, pred_jsons = gt_jsons['annotations'], pred_jsons['annotations']
    nframes_per_video = 6
    vid_num = len(gt_jsons) // nframes_per_video  # 600//6 = 100

    gt_pred_all = list(zip(gt_jsons, pred_jsons, gt_pans, pred_pans, gt_image_jsons))
    # array_split returns l % n sub-arrays of size l//n + 1 and the rest of size l//n.
    gt_pred_split = np.array_split(gt_pred_all, vid_num)

    start_time = time.time()
    vpq_all, vpq_thing, vpq_stuff, ids_errp = [], [], [], []
    vsq_all, vrq_all =[], []

    # for k in [0,5,10,15] --> num_frames_w_gt [1,2,3,4]
    ys_vpq, ys_vsq, ys_vrq, ys_errp = [], [], [], []
    cats_vpq, cats_vsq, cats_vrq, cats_errp = [], [], [], []
    all_pq_labels, all_sq_labels, all_rq_labels, all_errp_labels = [], [], [], []
    cats_pq_labels, cats_sq_labels, cats_rq_labels, cats_errp_labels = [], [], [], []
    # compute vpq for each k.
    for nframes in [1, 2, 3, 4]:
        gt_pred_split_ = copy.deepcopy(gt_pred_split)
        vsq_all_, vrq_all_, vpq_all_, vpq_thing_, vpq_stuff_, vpq_errp, th_errp, st_errp, ys_vpq, ys_vsq, ys_vrq, ys_errp, \
        cats_vpq, cats_vsq, cats_vrq, cats_errp = vpq_compute(
            gt_pred_split_, categories, nframes, output_dir, ys_vpq, ys_vsq, ys_vrq, ys_errp,
            cats_vpq, cats_vsq, cats_vrq, cats_errp)
        del gt_pred_split_
        print(vsq_all_, vrq_all_, vpq_all_, vpq_thing_, vpq_stuff_, vpq_errp)
        vpq_all.append(vpq_all_)
        vpq_thing.append(vpq_thing_)
        vpq_stuff.append(vpq_stuff_)
        ids_errp.append(vpq_errp)

        vsq_all.append(vsq_all_)
        vrq_all.append(vrq_all_)

        k = (nframes - 1) * 5
        metrics = [("All", None), ("Things", True), ("Stuff", False)]

        for name, _ in metrics:
            all_pq_labels.append(name + '_vpq_k_' + str(k))
            all_sq_labels.append(name + '_vsq_k_' + str(k))
            all_rq_labels.append(name + '_vrq_k_' + str(k))
            all_errp_labels.append(name + '_errp_k_' + str(k))
        cats_pq_labels.append('cats_vpq_k_' + str(k))
        cats_sq_labels.append('cats_vsq_k_' + str(k))
        cats_rq_labels.append('cats_vrq_k_' + str(k))
        cats_errp_labels.append('cats_errp_k_' + str(k))

    json.dump(cats_vpq, open(os.path.join(output_dir, 'vpq_cats.json'), 'w'))

    if args.draw_line_charts:
        x = list(range(len(gt_pred_split)))
        draw_line_charts(x, [ys_vpq, ys_vsq, ys_vrq, ys_errp],
                         [all_pq_labels, all_sq_labels, all_rq_labels, all_errp_labels], 360, 8.5, output_dir)
        draw_line_chart(cats_x, cats_vpq, cats_pq_labels, x_label='category',
                        y_label='cats_vpq', rotation=30, fontsize=8.5,
                        title='vpq_cats_fig', save_path=os.path.join(output_dir, 'vpq_cats_fig.png'))
        draw_line_chart(cats_x, cats_vsq, cats_sq_labels, x_label='category',
                        y_label='cats_vsq', rotation=30, fontsize=8.5,
                        title='vsq_cats_fig', save_path=os.path.join(output_dir, 'vsq_cats_fig.png'))
        draw_line_chart(cats_x, cats_vrq, cats_rq_labels, x_label='category',
                        y_label='cats_vrq', rotation=30, fontsize=8.5,
                        title='vrq_cats_fig', save_path=os.path.join(output_dir, 'vrq_cats_fig.png'))
        draw_line_chart(cats_x, cats_errp, cats_errp_labels, x_label='category',
                        y_label='cats_errp', rotation=30, fontsize=8.5,
                        title='errp_cats_fig', save_path=os.path.join(output_dir, 'errp_cats_fig.png'))

    print("vsq_all:%.4f\n" % (sum(vsq_all) / len(vsq_all)))
    print("vrq_all:%.4f\n" % (sum(vrq_all) / len(vrq_all)))
    print("vpq_all:%.4f\n" % (sum(vpq_all) / len(vpq_all)))
    print("vpq_thing:%.4f\n" % (sum(vpq_thing) / len(vpq_thing)))
    print("vpq_stuff:%.4f\n" % (sum(vpq_stuff) / len(vpq_stuff)))
    print("vpq_errp:%.4f\n" % (sum(ids_errp) / len(ids_errp)))
    print("------per-category vpq------:")
    per_value_list = []
    for i in range(len(cats_vpq[0])):
        per_value = 0
        for j in range(len(cats_vpq)):
            per_value += cats_vpq[j][i]
        per_value /= len(cats_vpq)
        per_value_list.append(per_value)
        print("category: {}, {}, average vpq: {}".format(cats_x[i], ' ' * (15 - len(cats_x[i])), str(per_value)[:5]))
    # print(per_value_list)

    output_filename = os.path.join(output_dir,str(curr_checkpoint)+ '_vpq-final.txt')
    output_file = open(output_filename, 'w')
    output_file.write("vpq_all:%.4f\n" % (sum(vpq_all) / len(vpq_all)))
    output_file.write("vpq_thing:%.4f\n" % (sum(vpq_thing) / len(vpq_thing)))
    output_file.write("vpq_stuff:%.4f\n" % (sum(vpq_stuff) / len(vpq_stuff)))
    output_file.write("vpq_errp:%.4f\n" % (sum(ids_errp) / len(ids_errp)))
    output_file.close()
    print('==> All:', time.time() - start_all, 'sec')



def parse_args():
    parser = argparse.ArgumentParser(description='VPSNet eval')
    parser.add_argument('--submit_dir', type=str,
        help='test outout directory', default='work_dirs/cityscapes_vps/fusetrack_vpct/val_pans_unified/') 
    parser.add_argument('--truth_dir', type=str, 
        help='ground truth directory', default='data/cityscapes_vps/val/panoptic_video')
    parser.add_argument('--pan_gt_json_file', type=str, 
        help='ground truth directory', default='data/cityscapes_vps/panpotic_gt_val_city_vps.json')
    parser.add_argument('--save_diff_fig', type=bool,
        help='save the pan_diff images', default=False)
    parser.add_argument('--draw_line_charts', type=bool,
                        help='save the pan_diff images', default=False)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    submit_dir = args.submit_dir
    truth_dir = args.truth_dir
    output_dir = submit_dir
    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)
    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # final_eval(args, submit_dir, truth_dir, output_dir)

    start_all = time.time()
    pan_pred_json_file = os.path.join(submit_dir, 'pred.json')
    with open(pan_pred_json_file, 'r') as f:
        pred_jsons = json.load(f)
    pan_gt_json_file = args.pan_gt_json_file
    with open(pan_gt_json_file, 'r') as f:
        gt_jsons = json.load(f)

    # gt_jsons keys: images, annotations, categories;  do not contain
    # pred_jsons keys: annotations

    cats_x = []
    for el in gt_jsons['categories']:
        cats_x.append(el['name'])
    print(cats_x)

    categories = gt_jsons['categories']
    categories = {el['id']: el for el in categories}
    # ==> pred_json, gt_json, categories
    # len(categories): 19.  id: {categories[i]}

    # load gt mask images to gt_pans and prediction mask images to pred_pans.
    start_time = time.time()
    gt_pans = []
    files = [
        item['file_name'].replace('_newImg8bit.png', '_final_mask.png').replace('_leftImg8bit.png', '_gtFine_color.png')
        for item in gt_jsons['images']]
    files.sort()
    # print(files)

    for idx, file in enumerate(files):
        image = np.array(Image.open(os.path.join(truth_dir, file)))
        gt_pans.append(image)
    print('==> gt_pans:', len(gt_pans), '//', time.time() - start_time, 'sec')

    start_time = time.time()
    pred_pans = []
    files = [item['id'] + '.png' for item in gt_jsons['images']]
    for idx, file in enumerate(files):
        image = np.array(Image.open(os.path.join(submit_dir, 'pan_pred', file)))
        pred_pans.append(image)
        if args.save_diff_fig:
            # generate the error map here.
            diff_pan = image - gt_pans[idx]
            diff_pan[np.where(gt_pans[idx] == 0)] = 0
            diff_image = Image.fromarray(diff_pan)
            os.makedirs(os.path.join(submit_dir, 'pan_diff'), exist_ok=True)
            diff_image.save(os.path.join(submit_dir, 'pan_diff', file))
    print('==> pred_pans:', len(pred_pans), '//', time.time() - start_time, 'sec')
    assert len(gt_pans) == len(pred_pans), "number of prediction does not match with the groud truth."

    gt_image_jsons = gt_jsons['images']
    gt_jsons, pred_jsons = gt_jsons['annotations'], pred_jsons['annotations']
    nframes_per_video = 6
    vid_num = len(gt_jsons) // nframes_per_video  # 600//6 = 100

    gt_pred_all = list(zip(gt_jsons, pred_jsons, gt_pans, pred_pans, gt_image_jsons))
    # array_split returns l % n sub-arrays of size l//n + 1 and the rest of size l//n.
    gt_pred_split = np.array_split(gt_pred_all, vid_num)

    start_time = time.time()
    vpq_all, vpq_thing, vpq_stuff, ids_errp = [], [], [], []
    vsq_all, vrq_all = [], []

    # for k in [0,5,10,15] --> num_frames_w_gt [1,2,3,4]
    ys_vpq, ys_vsq, ys_vrq, ys_errp = [], [], [], []
    cats_vpq, cats_vsq, cats_vrq, cats_errp = [], [], [], []
    all_pq_labels, all_sq_labels, all_rq_labels, all_errp_labels = [], [], [], []
    cats_pq_labels, cats_sq_labels, cats_rq_labels, cats_errp_labels = [], [], [], []
    # compute vpq for each k.
    for nframes in [1, 2, 3, 4]:
        gt_pred_split_ = copy.deepcopy(gt_pred_split)
        vsq_all_, vrq_all_, vpq_all_, vpq_thing_, vpq_stuff_, vpq_errp, th_errp, st_errp, ys_vpq, ys_vsq, ys_vrq, ys_errp, \
        cats_vpq, cats_vsq, cats_vrq, cats_errp = vpq_compute(
            gt_pred_split_, categories, nframes, output_dir, ys_vpq, ys_vsq, ys_vrq, ys_errp,
            cats_vpq, cats_vsq, cats_vrq, cats_errp)
        del gt_pred_split_
        print(vsq_all_, vrq_all_, vpq_all_, vpq_thing_, vpq_stuff_, vpq_errp)
        vpq_all.append(vpq_all_)
        vpq_thing.append(vpq_thing_)
        vpq_stuff.append(vpq_stuff_)
        ids_errp.append(vpq_errp)

        vsq_all.append(vsq_all_)
        vrq_all.append(vrq_all_)

        k = (nframes - 1) * 5
        metrics = [("All", None), ("Things", True), ("Stuff", False)]

        for name, _ in metrics:
            all_pq_labels.append(name + '_vpq_k_' + str(k))
            all_sq_labels.append(name + '_vsq_k_' + str(k))
            all_rq_labels.append(name + '_vrq_k_' + str(k))
            all_errp_labels.append(name + '_errp_k_' + str(k))
        cats_pq_labels.append('cats_vpq_k_' + str(k))
        cats_sq_labels.append('cats_vsq_k_' + str(k))
        cats_rq_labels.append('cats_vrq_k_' + str(k))
        cats_errp_labels.append('cats_errp_k_' + str(k))

    json.dump(cats_vpq, open(os.path.join(output_dir, 'vpq_cats.json'), 'w'))

    if args.draw_line_charts:
        x = list(range(len(gt_pred_split)))
        draw_line_charts(x, [ys_vpq, ys_vsq, ys_vrq, ys_errp],
                         [all_pq_labels, all_sq_labels, all_rq_labels, all_errp_labels], 360, 8.5, output_dir)
        draw_line_chart(cats_x, cats_vpq, cats_pq_labels, x_label='category',
                        y_label='cats_vpq', rotation=30, fontsize=8.5,
                        title='vpq_cats_fig', save_path=os.path.join(output_dir, 'vpq_cats_fig.png'))
        draw_line_chart(cats_x, cats_vsq, cats_sq_labels, x_label='category',
                        y_label='cats_vsq', rotation=30, fontsize=8.5,
                        title='vsq_cats_fig', save_path=os.path.join(output_dir, 'vsq_cats_fig.png'))
        draw_line_chart(cats_x, cats_vrq, cats_rq_labels, x_label='category',
                        y_label='cats_vrq', rotation=30, fontsize=8.5,
                        title='vrq_cats_fig', save_path=os.path.join(output_dir, 'vrq_cats_fig.png'))
        draw_line_chart(cats_x, cats_errp, cats_errp_labels, x_label='category',
                        y_label='cats_errp', rotation=30, fontsize=8.5,
                        title='errp_cats_fig', save_path=os.path.join(output_dir, 'errp_cats_fig.png'))

    print("vsq_all:%.4f\n" % (sum(vsq_all) / len(vsq_all)))
    print("vrq_all:%.4f\n" % (sum(vrq_all) / len(vrq_all)))
    print("vpq_all:%.4f\n" % (sum(vpq_all) / len(vpq_all)))
    print("vpq_thing:%.4f\n" % (sum(vpq_thing) / len(vpq_thing)))
    print("vpq_stuff:%.4f\n" % (sum(vpq_stuff) / len(vpq_stuff)))
    print("vpq_errp:%.4f\n" % (sum(ids_errp) / len(ids_errp)))

    print("------per-category vpq------:")
    per_value_list = []
    for i in range(len(cats_vpq[0])):
        per_value = 0
        for j in range(len(cats_vpq)):
            per_value += cats_vpq[j][i]
        per_value /= len(cats_vpq)
        per_value_list.append(per_value)
        print("category: {}, {}, average vpq: {}".format(cats_x[i], ' ' * (15 - len(cats_x[i])), str(per_value)[:5]))
    # print(per_value_list)

    output_filename = os.path.join(output_dir, 'vpq-final.txt')
    output_file = open(output_filename, 'w')
    output_file.write("vpq_all:%.4f\n" % (sum(vpq_all) / len(vpq_all)))
    output_file.write("vpq_thing:%.4f\n" % (sum(vpq_thing) / len(vpq_thing)))
    output_file.write("vpq_stuff:%.4f\n" % (sum(vpq_stuff) / len(vpq_stuff)))
    output_file.write("vpq_errp:%.4f\n" % (sum(ids_errp) / len(ids_errp)))
    output_file.close()
    print('==> All:', time.time() - start_all, 'sec')



if __name__ == "__main__":
    main()