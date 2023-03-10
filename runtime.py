checkpoint_config = dict(interval=1,
                        by_epoch=True,
                        save_optimizer=True,
                        out_dir='/home/data4/zjq/KAIST/pth',
                        max_keep_ckpts=2,
                        save_last=True)
# yapf:disable
log_config = dict(
    interval=100,
    by_epoch=False,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='CheckInvalidLossHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/zjq/ZYS/GAFF_mmdet/r18_gaff_fpn_3/best_bbox_mAP_50_epoch_29.pth'
resume_from = None  # '/home/data4/zjq/KAIST/pth/r18_gaff_fpn_2/epoch_11.pth'
work_dir = './GAFF_mmdet/r18_gaff_fpn_1'

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (1 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(base_batch_size=8) # enable=False, 
