checkpoint_config = dict(interval=1,
                        by_epoch=True,
                        save_optimizer=True,
                        out_dir=None,
                        max_keep_ckpts=7,
                        save_last=True)
# yapf:disable
log_config = dict(
    interval=100,
    by_epoch=False,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
work_dir = None

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (1 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(base_batch_size=8) # enable=False, 
