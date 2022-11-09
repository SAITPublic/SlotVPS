import argparse
import os
import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from tools.config.config import config, update_config
import pickle
import json
from tools.test_vpq import single_gpu_test
from tools.eval_vpq import final_eval
from mmdet.utils import get_model_parameters_number, params_to_string
from tools.dataset import *


def parse_args():
    parser = argparse.ArgumentParser(description='VPSNet test')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint_dir', help='checkpoint file dir path')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--load', help='when .pkl files are saved',
        action='store_true')
    parser.add_argument('--gpus', type=str, default='0' )
    parser.add_argument('--dataset', type=str, default='CityscapesVps')
    parser.add_argument('--test_config', type=str,
        default='configs/cityscapes/test_cityscapes_1gpu.yaml')
    # ---- VPQ - specific arguments
    parser.add_argument('--n_video', type=int, default=50)
    parser.add_argument('--pan_im_json_file', type=str, default='data/cityscapes_vps/panoptic_im_val_city_vps.json')
    # ==== for eval process
    parser.add_argument('--truth_dir', type=str,
                        help='ground truth directory', default='data/cityscapes_vps/val/panoptic_video')
    parser.add_argument('--pan_gt_json_file', type=str,
                        help='ground truth directory', default='data/cityscapes_vps/panpotic_gt_val_city_vps.json')
    parser.add_argument('--save_diff_fig', type=bool,
                        help='save the pan_diff images', default=False)
    parser.add_argument('--draw_line_charts', type=bool,
                        help='draw line charts', default=False)
    parser.add_argument('--mode', type=str, default='val',
                        help='denote test the val split or the test split')
    parser.add_argument('--only_eval_pq', help='whether to only calculate the pq',
                        action='store_true')
    parser.add_argument('--generate_segms_new_version', help='whether to use the new version to generate segms',
                        action='store_true')
    parser.add_argument('--eval_by_video', help='whether to test video by video', action='store_true')
    parser.add_argument('--only_test_frames_labeled', help='whether only test the frames labeled', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    args, rest = parser.parse_known_args()
    update_config(args.test_config)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    gpus = [int(_) for _ in args.gpus.split(',')]
    if args.out is not None and not args.out.endswith(('.pkl', 'pickle')):
        raise ValueError("The output file must be a .pkl file.")

    cfg = mmcv.Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backedns.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    distributed = False

    if args.mode == 'val':
        cfg.data.test.ann_file = cfg.data.test.ann_file.replace("test", "val")
        cfg.data.test.img_prefix = cfg.data.test.img_prefix.replace("test", "val")
        cfg.data.test.ref_prefix = cfg.data.test.ref_prefix.replace("test", "val")
    elif args.mode == 'test':
        cfg.data.test.ann_file = cfg.data.test.ann_file.replace("val", "test")
        cfg.data.test.img_prefix = cfg.data.test.img_prefix.replace("val", "test")
        cfg.data.test.ref_prefix = cfg.data.test.ref_prefix.replace("val", "test")
    else:
        raise KeyError("Argument 'mode' must be either 'val' or 'test'.")

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # update the args.out to cfg
    if "other_config" in cfg.model:
        cfg.model.other_config['output_dir'] = args.out
    if 'postprocess_panoptic' in cfg.model:
        cfg.model.postprocess_panoptic['output_dir'] = args.out
    print(cfg)
    # build the model and load checkpoint
    model = build_detector(cfg.model,
                           train_cfg=None,
                           test_cfg=cfg.test_cfg)
    print(model)

    print("========================")
    print("Model Params : {}".format(params_to_string(get_model_parameters_number(model))))
    print("========================")

    checkpoint = load_checkpoint(model,
                                 args.checkpoint_dir,
                                 map_location='cpu')

    # E.g., Cityscapes has 8 things CLASSES.
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    eval_helper_dataset = eval(args.dataset)()

    output_dir = args.out.replace('.pkl', '_pans_unified/')
    print("==> Video Panoptic Segmentation results will be saved at:")
    print("---", output_dir)

    with open(args.pan_im_json_file, 'r') as f:
        im_jsons = json.load(f)
    names = [x['file_name'] for x in im_jsons['images']]
    names.sort()
    categories = im_jsons['categories']

    if args.load:
        pans_2ch_pkl = args.out.replace('.pkl', '_pred_pans_2ch.pkl')
        pred_pans_2ch = pickle.load(open(pans_2ch_pkl, 'rb'))
    else:
        model = MMDataParallel(model, device_ids=[gpus[0]])

        outputs_pano = single_gpu_test(model, data_loader)

        obj_ids = outputs_pano['all_pano_obj_ids']
        pred_pans_2ch_ = eval_helper_dataset.get_unified_pan_result(
            outputs_pano['all_ssegs'],
            outputs_pano['all_panos'],
            outputs_pano['all_pano_cls_inds'],
            obj_ids=obj_ids,
            stuff_area_limit=config.test.panoptic_stuff_area_limit,
            names=outputs_pano['all_names'])
        print('==> Done: vps_unified_pan_result')

        pred_keys = [_ for _ in pred_pans_2ch_.keys()]
        pred_keys.sort()  # # 'frankfurt_000000_001732_leftImg8bit.png', 'frankfurt_000000_001733_leftImg8bit.png'
        # note that the pred_keys is not same as the "names". # 0005_0025_frankfurt_000000_001736_newImg8bit.png
        pred_pans_2ch = [pred_pans_2ch_[k] for k in pred_keys]
        del pred_pans_2ch_
        with open(args.out.replace('.pkl', '_pred_pans_2ch.pkl'), 'wb') as f:
            pickle.dump(pred_pans_2ch, f, protocol=2)

    if args.dataset == 'Viper':
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        results = eval_helper_dataset.evaluate_panoptic(pred_pans_2ch, output_dir, n_video=args.n_video,
                                                        save_name=output_dir,
                                                        only_calculate_pq=args.only_eval_pq,
                                                        generate_segms_new_version=args.generate_segms_new_version)
    else:
        assert args.dataset == 'CityscapesVps'
        # sample 500 results here.
        pred_pans, pred_json = eval_helper_dataset.inference_panoptic_video(
            pred_pans_2ch, output_dir,
            categories=categories,
            names=names,
            # pan_im_json_file=args.pan_im_json_file,
            n_video=args.n_video)
        print('==> Done: vps_inference of checkpoint ' + str(args.checkpoint_dir))

        if args.pan_gt_json_file != 'None':
            # start eval process
            truth_dir = args.truth_dir
            if os.path.isdir(output_dir) and os.path.isdir(truth_dir):
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

            print('start evaling for ' + output_dir + '.....')
            try:
                final_eval(args, output_dir, args.truth_dir, output_dir)
            except ZeroDivisionError:
                print('zero division error ....')

        del pred_pans, pred_json

    torch.cuda.empty_cache()
    del model, checkpoint
    torch.cuda.empty_cache()

    # remove all .pkl file.
    os.remove(args.out.replace('.pkl', '_pred_pans_2ch.pkl'))


if __name__ == '__main__':
    main()