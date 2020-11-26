_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
# 建立了软连接 s2_detection -> /home/liuyuan/shu_codes/competitions/fake_image_detection/codes_shu/
data_root = '/home/liuyuan/shu_codes/mmdetection/s2_data_with_bonus_coco'
# We also need to change the num_classes in head to match the dataset's annotation
# 训练相关
import torchvision

classes = ('tampered',)
num_classes = len(classes)
# optimizer = dict(type='SGD', lr=1e-4, weight_decay=0.0001, momentum=0.9)
optimizer = dict(type='Adam', lr=1e-4, weight_decay=0.0001, _delete_=True)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-8, by_epoch=False, _delete_=True)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[16, 22, 32, 46, 61, 75, 87])
# Mask RCNN 模型
model = dict(
    type='RGBNMaskRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    # 不知道能否共享权重？
    noise_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    noise_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='RGBNStandardRoIHead',
        noise_roi_extractor=dict(
            # 应该是按照 BBOX 的 ROI
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            # shared 2 FCs
            # type='RGBNShared2FCBBoxHead',
            # in_channels=256,
            # fc_out_channels=1024,
            # roi_feat_size=7,
            # num_classes=num_classes,
            # bbox_coder=dict(
            #     type='DeltaXYWHBBoxCoder',
            #     target_means=[0., 0., 0., 0.],
            #     target_stds=[0.1, 0.1, 0.2, 0.2]),
            # reg_class_agnostic=False,
            # loss_cls=dict(
            #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            # loss_bbox=dict(type='L1Loss', loss_weight=1.0)
            # 分开 reg 和 cls
            type='RGBNBBoxHead',
            in_channels=256,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            with_avg_pool=False,
            pooling_size=512,
            _delete_=True
            ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))))

# 数据流的处理, to_rgb 表示间读取的图像从 BGR 到 RGB
# img_norm_cfg = dict(
#     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1600, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=256),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=256),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# 数据集
# TODO 注意， 这里的 category_id = 0 了，
dataset_type = 'COCODataset'
data = dict(
    train=dict(
        data_root=data_root,
        img_prefix='train_image',
        classes=classes,
        pipeline=train_pipeline,
        ann_file=f'{data_root}/train_rate0.9/instances_train.json'),
    val=dict(
        # 如果 val 数据有 annotation 等信息就会自动地加载, 因此无需担心没有加载 ann: https://github.com/open-mmlab/mmdetection/issues/1401
        data_root=data_root,
        img_prefix='train_image',
        pipeline=test_pipeline,
        classes=classes,
        ann_file=f'{data_root}/train_rate0.9/instances_validation.json'),
    test=dict(
        data_root=data_root,
        img_prefix='test_image',
        pipeline=test_pipeline,
        # classes=classes,
        ann_file=f'{data_root}/instances_test.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = '/home/liuyuan/shu_codes/competitions/fake_image_detection/detection_test/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
# load_from = 'https://open-mmlab.oss-cn-beijing.aliyuncs.com/mmdetection/models/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth'
# load_from = 'http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
workflow = [('train', 1), ('val', 1)]
total_epochs = 50
# load_from = 'work_dirs/mask_rcnn/latest.pth'
work_dir = './work_dirs/s2_sgd_rgbn_mask_rcnn_r50_fpn_1x'
