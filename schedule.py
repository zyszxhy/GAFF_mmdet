# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     by_epoch=False,
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.0001,
#     warmup_by_epoch=False,
#     min_lr=1e-6)
# runner = dict(type='IterBasedRunner', max_iters=3500)

# workflow = [('train', 100)]
evaluation = dict(start=3, interval=1, metric='bbox')

optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    # warmup='linear',
    # warmup_iters=500,
    # warmup_ratio=0.001,
    gamma=0.1,
    step=[6,12])
workflow = [('train', 1)]
runner = dict(type='EpochBasedRunner', max_epochs=18)