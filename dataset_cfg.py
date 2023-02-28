# dataset settings
custom_imports = dict(
    imports=[# 'mmdet.datasets.kaist',
    'mmdet.datasets.pipelines.kaist_pipelines',
    'mmdet.models.necks.gaff_blocks',
    'mmdet.models.detectors.gaff_detector'],
    allow_failed_imports=False)
# custom_imports = dict(
#     imports=['..GAFF.custom_files.kaist',
#     '..GAFF.custom_files.kaist_pipelines',
#     '..GAFF.custom_files.gaff_blocks',
#     '..GAFF.custom_files.gaff_detector'],
#     allow_failed_imports=False)

# dataset_type = 'KAISTDataset'
dataset_type = 'CocoDataset'
classes = ('person',)
data_root = './autodl-tmp/kaist/'
img_norm_cfg = dict(
    mean_visible=[123.675, 116.28, 103.53], std_visible=[1, 1, 1], \
    mean_lwir=[135.438, 135.438, 135.438], std_lwir=[1, 1, 1], \
    to_rgb=True)
train_pipeline = [
    dict(type='LoadKAISTImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True ,with_mask=True, poly2mask=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomCropWithProbility', crop_size=(0.3, 1.0), crop_type='my_crop', \
        allow_negative_crop=False, crop_ratio=0.5),
    dict(type='Resize', img_scale=(640, 512), keep_ratio=False),
    dict(type='KAISTNormalize', **img_norm_cfg),
    dict(type='KAISTPhotoMetricDistortion', \
        brightness_range=(0.5, 2.0), saturation_range=(0.75, 1.5), hue_range=(0.75, 1.5)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']), # 
]
test_pipeline = [
    dict(type='LoadKAISTImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='KAISTNormalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'kaist_train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'kaist_test',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'kaist_test',
        pipeline=test_pipeline))
