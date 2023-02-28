# from mmdet.models.builder import build_backbone, build_neck, build_head, build_detector

# import torch
# import torch.nn as nn

# checkpoint_path = './checkpoints/retinanet_r18_fpn_1x_coco_20220407_171055-614fd399.pth'
# pre_weights_dict = torch.load(checkpoint_path)
# print(pre_weights_dict)
# backbone_visible=dict(
#         type='ResNet',
#         depth=18,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=0,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         norm_eval=True,
#         style='pytorch',
#         init_cfg=dict(type='Pretrained', checkpoint=checkpoint_path, prefix='backbone.'))

# model_backbone = build_backbone(backbone_visible)
# model_backbone.init_weights()
# for name, parameters in model_backbone.named_parameters():  
#     print(name, ';', parameters.size())

# class GAFF(nn.Module):
#     def __init__(self, backbone, neck, bbox_head, train_cfg, test_cfg):
#         super(GAFF, self).__init__()
#         self.Backbone = build_backbone(backbone)
#         self.Neck = build_neck(neck)
#         bbox_head.update(train_cfg=train_cfg)
#         bbox_head.update(test_cfg=test_cfg)
#         self.bbox_head = build_head(bbox_head)

#     def forward(self, x, x_lwir):
#         x, x_lwir = self.Backbone(x, x_lwir)
#         outs, masks = self.Neck(x, x_lwir)
#         cls_scores, bbox_pred = self.bbox_head(outs)
#         return tuple(cls_scores), tuple(bbox_pred), tuple(masks)


# from mmcv import Config
# from mmdet.utils import (replace_cfg_vals, update_data_root)
# cfg = Config.fromfile('./model_cfg_1.py')
# # cfg = replace_cfg_vals(cfg)
# # update_data_root(cfg)
# # model = GAFF(backbone=cfg.model.backbone, neck=cfg.model.neck, bbox_head=cfg.model.bbox_head, \
# #                 train_cfg=cfg.model.train_cfg, test_cfg=cfg.model.test_cfg)
# model = build_detector(
#         cfg.model,
#         train_cfg=cfg.get('train_cfg'),
#         test_cfg=cfg.get('test_cfg'))
# model.eval()

# from tensorboardX import SummaryWriter
# writer = SummaryWriter('./runs')
# img = torch.rand(3, 3, 512, 640)
# img_lwir = torch.rand(3, 3, 512, 640)
# cls_scores, bbox_pred, masks = model(img, img_lwir)
# print(cls_scores)
# print(bbox_pred)
# print(masks)
# input = (img, img_lwir)
# writer.add_graph(model, input)
# writer.close()


# from mmdet.datasets import CocoDataset
# from mmdet.datasets.pipelines import LoadKAISTImageFromFile, LoadAnnotations, RandomCropWithProbility,\
#     RandomFlip, Resize, KAISTNormalize, KAISTPhotoMetricDistortion, \
#         DefaultFormatBundle, Collect



# dataset_type = 'CocoDataset'
# classes = ('person')
# data_root = './GAFF/KAIST'
# img_norm_cfg = dict(
#     mean_visible=[123.675, 116.28, 103.53], std_visible=[1, 1, 1], \
#     mean_lwir=[135.438, 135.438, 135.438], std_lwir=[1, 1, 1], \
#     to_rgb=True)
# ann_file= 'annotations/test.json'
# img_prefix= 'kaist_test/kaist_test'

# dataset_test = CocoDataset(ann_file=ann_file, pipeline='', \
#     data_root=data_root, img_prefix=img_prefix, classes=classes)
# print(dataset_test.__len__())
# for i in range(dataset_test.__len__()):
#     img_info = dataset_test.data_infos[i]
#     ann_info = dataset_test.get_ann_info(i)
#     results = dict(img_info=img_info, ann_info=ann_info)
#     results['img_prefix'] = dataset_test.img_prefix
#     results['seg_prefix'] = dataset_test.seg_prefix
#     results['proposal_file'] = dataset_test.proposal_file
#     results['bbox_fields'] = []
#     results['mask_fields'] = []
#     results['seg_fields'] = []

#     loadimg = LoadKAISTImageFromFile()
#     loadann = LoadAnnotations(with_bbox=True, with_mask=True, poly2mask=True)
#     flipimg = RandomFlip(flip_ratio=0.5, direction='horizontal')
#     cropimg = RandomCropWithProbility(crop_size=(0.3, 1.0), crop_type='my_crop', \
#             allow_negative_crop=False, crop_ratio=0.5)
#     resizeimg = Resize(img_scale=(640, 512), keep_ratio=False)
#     normalizeimg = KAISTNormalize(**img_norm_cfg)
#     photodistor = KAISTPhotoMetricDistortion(brightness_range=(0.5, 2.0), saturation_range=(0.75, 1.5), hue_range=(0.75, 1.5))
#     format = DefaultFormatBundle()
#     collect = Collect(keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])

#     results = loadimg(results)
#     print(results['img_shape'])
#     results = loadann(results)
#     print(results['img_shape'])
#     results = flipimg(results)
#     print(results['img_shape'])
#     results = cropimg(results)
#     # print(results['img_shape'])
#     results = resizeimg(results)
#     print(results['img_shape'])
#     results = normalizeimg(results)
#     print(results['img_shape'])
#     results = photodistor(results)
#     print(results['img_shape'])
#     results = format(results)
#     print(results['img_shape'])
#     results = collect(results)
#     print(results['img_metas'].data['img_shape'])

# import numpy as np
# import cv2
# import torch.nn.functional as F
# import torch
# height = 64
# width = 80
# center_x = 25
# center_y = 20
# box_width = 15
# box_height = 25
# mask = np.zeros((height, width), dtype=np.uint8)
# cv2.ellipse(mask, (center_x, center_y), (int(np.ceil(box_width/2)), int(np.ceil(box_height/2))), 0, 0, 360, 1, -1)                     
# mask_1=cv2.flip(mask, 1)
# masks = torch.FloatTensor([mask, mask_1])
# masks_pre = torch.FloatTensor([mask_1, mask_1])
# masks = masks[None, :, :, :]
# masks_pre = masks_pre.unsqueeze(0)
# from mmdet.models.losses import DiceLoss
# dice_loss = DiceLoss(
#     use_sigmoid=True,
#     activate=True,
#     reduction='mean',
#     naive_dice=False,
#     loss_weight=1.0,
#     eps=1e-3)
# loss = dice_loss(masks, masks_pre)
# print(loss)
# mask_up = F.interpolate(masks, size=(512, 640), mode='bilinear')

# import matplotlib.pyplot as plt
# plt.imshow(mask)
# plt.show()
# plt.imshow(mask_1)
# plt.show()
# plt.imshow(mask_up[0,0,:,:])
# plt.show()
# plt.imshow(mask_up[1,0,:,:])
# plt.show()

# import numpy as np
# masks = np.load('./mask.npy')
# import matplotlib.pyplot as plt
# plt.imshow(masks[0][0])
# plt.show()
# plt.imshow(masks[1][0])
# plt.show()
# plt.imshow(masks[2][0])
# plt.show()

# import cv2
# import numpy as np

# width = 640
# height = 512
# mask = np.zeros((height, width), dtype=np.uint8)
# contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# contours = list(contours)
# contours = [cor.reshape(-1) for cor in contours]
# polygons = []
# polygons.append(contours)

# import torch
# checkpoint_path = './checkpoints/retinanet_r18_fpn_1x_coco_20220407_171055-614fd399.pth'
# pre_weights_dict = torch.load(checkpoint_path)
# print(pre_weights_dict)