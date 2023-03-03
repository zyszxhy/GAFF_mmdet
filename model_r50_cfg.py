checkpoint_path = './GAFF_mmdet/ckps/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'

model = dict(
    type='GAFFDetector',
    backbone_visible=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_path, prefix='backbone.')
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        ),
    backbone_lwir=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_path, prefix='backbone.')
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        ),

    neck_visible=dict(
        type='FPN',
        # ResNet 模块输出的4个尺度特征图通道数
        in_channels=[256, 512, 1024, 2048],
        # FPN 输出的每个尺度输出特征图通道
        out_channels=256,
        # 从输入多尺度特征图的第几个开始计算
        start_level=1,
        # 额外输出层的特征图来源
        add_extra_convs='on_input',
        # FPN 输出特征图个数
        num_outs=5,
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_path, prefix='neck.')
        init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')
        ),
    neck_lwir=dict(
        type='FPN',
        # ResNet 模块输出的4个尺度特征图通道数
        in_channels=[256, 512, 1024, 2048],
        # FPN 输出的每个尺度输出特征图通道
        out_channels=256,
        # 从输入多尺度特征图的第几个开始计算
        start_level=1,
        # 额外输出层的特征图来源
        add_extra_convs='on_input',
        # FPN 输出特征图个数
        num_outs=5,
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_path, prefix='neck.')
        init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')
        ),

    gaff=dict(
        type='GAFF',
        num_ins=5,
        in_channels=[256, 256, 256, 256, 256],
        init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')
    ),

    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
                type='AnchorGenerator',
                scales=[0.5, 1, 1.5],
                base_sizes=[27, 36, 49, 66, 130],
                # ratios=[0.41],
                ratios=[2.44],
                # octave_base_scale=4,
                # scales_per_octave=3,
                # ratios=[1.0, 2.0],
                strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
        loss_bbox=dict(type='BalancedL1Loss', loss_weight=1.0),
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_path, prefix='bbox_head.')
        init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')
        ),
    
    loss_intra=dict(
        type='DiceLoss',
        activate=False,
        reduction='mean',
        naive_dice=False,
        loss_weight=0.1,
        eps=1e-3),
    
    loss_inter=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        use_mask=False,
        reduction='mean',
        class_weight=None,
        ignore_index=2,
        loss_weight=1.0,
        avg_non_ignore=False
    ),

    train_cfg=dict(
        assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,# 0.7
                neg_iou_thr=0.4,# 0.5
                min_pos_iou=0,
                ignore_iof_thr=-1),# 0.5
        allowed_border=-1,
        pos_weight=-1,
        debug=False),

    test_cfg=dict(
        nms_pre=1000,# 6000
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
