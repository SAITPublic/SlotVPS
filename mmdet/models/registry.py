from mmdet.utils import Registry

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
PANOPTIC = Registry('panoptic')
HEADS = Registry('head')
LOSSES = Registry('loss')
DETECTORS = Registry('detector')
