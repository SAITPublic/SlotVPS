# model settings
model = dict(
    type= 'VPS_Temporal_Slots',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        embed_dim=192,#96,
        depths=[2, 2, 18, 2], #[2, 2, 6, 2],
        num_heads=[6, 12, 24, 48],#[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.5, #0.2,  # follow the setting of panoptic-fcn swin based config.
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536],
        out_channels=256,
        num_outs=5),
    panoptic = dict(
            type='UPSNetFPN',
            in_channels=256,
            out_channels=128,
            num_levels=4,
            num_things_classes=8,
            num_classes=19,
            ignore_label=255,
            loss_weight=0.5),
    dynamic_mask_head=dict(
        dh_dim=256,
        num_classes=20,  # 11 + 8 + 1
        dim_feedforward=2048,
        nhead=8,
        dropout=0.0,
        activation="relu",
        dh_num_heads=7,
        per_dh_num_heads=[1, 2, 2, 2],
        feat_num_levels=4,
        merge_operation="concat",
        trans_in_dim=384,
        return_intermediate=True,
        use_focal=True,
        prior_prob=0.01,
        num_cls=2,
        num_reg=2,
        temporal_query_attention_config=dict(
            d_model=256,
            dim_feedforward=1024,
            dropout=0.0,
            activation="gelu",
            softmax_dim="slots",
            drop_path=0.,
        ),
        apply_temporal_query_atten_stages=[3, 4, 5, 6],
    ),
    # maxdeeplablossC=dict(
    #     num_classes=20,
    #     pq_loss_weight=3,
    #     instance_loss_weight=1,
    #     maskid_loss_weight=0.3,
    #     alpha=0.75,
    #     temp=0.3,
    #     class_loss_option='binary_cross_entropy',
    #     mask_id_loss_option='cross_entropy',
    #     insdis_loss_option='hand_craft',
    # ),
    postprocess_panoptic=dict(
        is_thing_map={i: i > 10 for i in range(20)},
        threshold=0.85,
        fraction_threshold=0.03,
        pixel_threshold=0.4,
        apply_mask_removal=True,
        apply_mask_removal_only_ins=True,
        use_mask_low_constant=False,
    ),
    # panoptic_clip_matcher=dict(
    #     num_classes=20,
    #     semantic_loss_weight=0.5,
    #     maxdeeplablossC_config=dict(
    #         num_classes=20,
    #         pq_loss_weight=3,
    #         instance_loss_weight=1,
    #         maskid_loss_weight=0.3,
    #         alpha=0.75,
    #         temp=0.3,
    #         class_loss_option='binary_cross_entropy',
    #         mask_id_loss_option='cross_entropy',
    #         insdis_loss_option='hand_craft',
    #     ),
    # ),
    simple_track_head=dict(
        loss_match=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5),
        num_fcs_query=2,
        in_channels_query=256,
        query_matched_weight=1.0,
    ),
    other_config=dict(
        proposal_num=100,
        has_no_obj=True,  # align with num_classes
        pos_config=dict(
            position_embedding="sine",  # ("sine", "learned")
            hidden_dim=256,
        ),
        test_forward_ref_img=True,
        test_only_save_main_results=True,
        ),
)

# model training and testing settings
train_cfg = dict(
    loss_pano_weight=0.5,
    class_mapping = {1:11, 2:12, 3:13, 4:14, 5:15, 6:16, 7:17, 8:18},
)
test_cfg = dict(
    loss_pano_weight=None,
    class_mapping = {1:11, 2:12, 3:13, 4:14, 5:15, 6:16, 7:17, 8:18},
)
# dataset settings
dataset_type = 'CityscapesVPSDataset'
data_root = 'data/cityscapes_vps/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadRefImageFromFile', with_pad_mask=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, 
        with_seg=True, with_pid=True,
        # Cityscapes specific class mapping
        semantic2label={0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9,
                        10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16,
                        17:17, 18:18, -1:255, 255:255},),
    dict(type='Resize', img_scale=[(2048, 1024)], keep_ratio=True,
        multiscale_mode='value', ratio_range=(0.8, 1.5)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomCrop', crop_size=(800, 1600)),  # (800, 1600)
    dict(type='Pad', size_divisor=32),
    dict(type='SegResizeFlipCropPadRescale', scale_factor=[1, 0.25]),
    dict(type='FixedImageRandomShift'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 
                               'gt_obj_ids', 'gt_masks', 'gt_semantic_seg',
                               'gt_semantic_seg_Nx','ref_img', 'ref_bboxes',
                               'ref_labels', 'ref_obj_ids', 'ref_masks',
                               'pad_mask', 'ref_pad_mask',
                                'ref_semantic_seg',]),    # 'ref_semantic_seg_Nx'
]
test_pipeline = [
    dict(type='LoadRefImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(2048, 1024)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'ref_img']),
            dict(type='Collect', keys=['img', 'ref_img']),
        ])
]
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        # times=1,
        times=8,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root +
            'instances_train_city_vps_rle.json',
            img_prefix=data_root + 'train/img/',
            ref_prefix=data_root + 'train/img/',
            seg_prefix=data_root + 'train/labelmap/',
            pipeline=train_pipeline,
            ref_ann_file=data_root + 
            'instances_train_city_vps_rle.json',
            offsets='0_shift_3')),  # [-1,+1]  # "all"
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        'instances_val_city_vps_rle.json',
        img_prefix=data_root + 'val/img/',
        ref_prefix=data_root + 'val/img/',  # new added
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        'im_all_info_val_city_vps.json',
        img_prefix=data_root + 'val/img_all/',
        ref_prefix=data_root + 'val/img_all/',
        nframes_span_test=30,
        pipeline=test_pipeline))
# optimizer
# lr is set for a batch size of 8
# optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=1., norm_type=2))
# learning policy
lr_config = dict(
    policy='step',   # 'cosine',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11],
    min_lr_ratio=0.01,
)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
# runtime settings
total_epochs = 12 # 16
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/cityscapes_vps/track_vpct'
# load_from = './work_dirs/viper/track/latest.pth'
load_from = './work_dirs/cityscapes_vps/track_capsule_stg6_100_c20_dhifuse_noft_nofb_8001600_e12_05_d256_possine_dgm_ms1222afdcat256v0_p816_nr640_newlff2ebusefbch_lrddb_swinL_v100g8/epoch_10_rename.pth'
# resume_from = './work_dirs/cityscapes/ups_async_cococity_512x2/latest.pth'
resume_from = None
workflow = [('train', 1)]
