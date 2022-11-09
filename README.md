# Slot-VPS: Object-centric Representation Learning for Video Panoptic Segmentation

Official implementation for ["Slot-VPS: Object-centric Representation Learning for Video Panoptic Segmentation"](https://arxiv.org/abs/2112.08949) (CVPR 2022)

## Installation
a. This repo is built based on [mmdetection](https://github.com/open-mmlab/mmdetection) commit hash `4357697`. Our modifications for Slot-VPS implementation are listed [here](mmdet/readme.txt). Note that except torch1.9.0+cu111, we also validated that it can work with torch1.7.0+cu110 or torch1.4.0.
You can use following commands to create conda env with related dependencies. Take torch1.9.0+cu111 as example:
```
conda create -n slotvps
conda activate slotvps
pip install -r requirements.txt
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install "git+https://github.com/cocodataset/panopticapi.git"
pip install -v -e .
```

## Dataset
Follow the [instructions](https://github.com/mcahny/vps#dataset) to prepare Cityscapes-VPS and VIPER datasets.


## Testing
Run the following commands to test the model on Cityscapes-VPS. Note that now the pth files are unavailable. We are trying to find ways to upload large files.

* Slot-VPS model for Video Panoptic Quality (VPQ) on Cityscapes-VPS `val` set (`vpq-λ.txt` will be saved.)
```
python ./tools/test_eval_vpq.py \
--config configs/cityscapes/r50_fpn_slotvps.py \
--checkpoint_dir ./work_dirs/cityscapes_vps/r50_fpn_slotvps/r50_fpn_slotvps.pth \
--out ./work_dirs/cityscapes_vps/r50_fpn_slotvps/val_test.pkl \
--pan_im_json_file data/cityscapes_vps/panoptic_im_val_city_vps.json \
--mode 'val' \
--n_video 50 \
--truth_dir data/cityscapes_vps/val/panoptic_video/ \
--pan_gt_json_file data/cityscapes_vps/panoptic_gt_val_city_vps.json \
```
* Slot-VPS model VPS inference on Cityscapes-VPS `test` set
```
python ./tools/test_eval_vpq.py \
--config configs/cityscapes/r50_fpn_slotvps.py \
--checkpoint_dir ./work_dirs/cityscapes_vps/r50_fpn_slotvps/r50_fpn_slotvps.pth \
--out ./work_dirs/cityscapes_vps/r50_fpn_slotvps/test.pkl \
--pan_im_json_file data/cityscapes_vps/panoptic_im_test_city_vps.json \
--mode 'test' \
--n_video 50 \
--truth_dir data/cityscapes_vps/val/panoptic_video/ \
--pan_gt_json_file 'None' \
```
Files containing the predicted results will be generated as `pred.json` and `pan_pred/*.png` at  `work_dirs/cityscapes_vps/fusetrack_vpct/test_pans_unified/`.

Cityscapes-VPS `test` split currently only allows evaluation on the codalab server. Please upload `submission.zip` to <a href="https://competitions.codalab.org/competitions/26183">Codalab server</a> to see actual performances.
```
submission.zip
├── pred.json
├── pan_pred.zip
│   ├── 0005_0025_frankfurt_000000_001736.png
│   ├── 0005_0026_frankfurt_000000_001741.png
│   ├── ...
│   ├── 0500_3000_munster_000173_000029.png
```


## Training
Currently, we do not provide the code for training.


## Citation

If you use this toolbox or benchmark in your research, please cite this project.  # may need to update the name.

```bibtex
@inproceedings{zhou2021slot,
  title={Slot-VPS: Object-centric Representation Learning for Video Panoptic Segmentation},
  author={Zhou, Yi and Zhang, Hui and Lee, Hana and Sun, Shuyang and Li, Pingjun and Zhu, Yangguang and Yoo, ByungIn and Qi, Xiaojuan and Han, Jae-Joon},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Terms of Use

This software is for non-commercial use only.


## Acknowledgements
This project has used utility functions from other wonderful open-sourced libraries. We would especially thank the authors of:
* [mmdetection](https://github.com/open-mmlab/mmdetection)
* [VPSNet](https://github.com/mcahny/vps)


## Contact

If you have any questions regarding the repo, please contact Yi ZHOU (yi0813.zhou@samsung.com) or create an issue.
 
