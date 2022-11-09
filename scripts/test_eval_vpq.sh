python -u ./tools/test_eval_vpq.py \
--config configs/cityscapes/r50_fpn_slotvps.py \
--checkpoint_dir ./work_dirs/cityscapes_vps/r50_fpn_slotvps/r50_fpn_slotvps.pth \
--out ./work_dirs/cityscapes_vps/r50_fpn_slotvps/val.pkl \
--pan_im_json_file data/cityscapes_vps/panoptic_im_val_city_vps.json \
--mode 'val' \
--n_video 50 \
--truth_dir data/cityscapes_vps/val/panoptic_video/ \
--pan_gt_json_file data/cityscapes_vps/panoptic_gt_val_city_vps.json \
2>&1 |tee test_Slotvps.log
