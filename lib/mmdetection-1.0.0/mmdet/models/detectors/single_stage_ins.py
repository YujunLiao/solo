import os

import torch
import torch.nn as nn
import torchvision

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector


@DETECTORS.register_module
class SingleStageInsDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_feat_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageInsDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        if mask_feat_head is not None:
            self.mask_feat_head = builder.build_head(mask_feat_head)

        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageInsDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_mask_feat_head:
            if isinstance(self.mask_feat_head, nn.Sequential):
                for m in self.mask_feat_head:
                    m.init_weights()
            else:
                self.mask_feat_head.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        if self.with_mask_feat_head:
            mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                  start_level:self.mask_feat_head.end_level + 1])
            loss_inputs = outs + (mask_feat_pred, gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x, eval=True)

        if self.with_mask_feat_head:
            mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                  start_level:self.mask_feat_head.end_level + 1])
            seg_inputs = outs + (mask_feat_pred, img_meta, self.test_cfg, rescale)
        else:
            seg_inputs = outs + (img_meta, self.test_cfg, rescale)
        seg_result = self.corruput_seg_result(img_meta, self.test_cfg, ori_seg_inputs = seg_inputs)
        # seg_result = self.bbox_head.get_seg(*seg_inputs)
        # self.save_seg_result(seg_result, img_meta)
        return seg_result

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    def corruput_seg_result(self, img_meta, test_cfg, ori_seg_inputs):
        # # get cate_pred
        # cate_pred = []
        # for size in [40, 36, 24, 16, 12]:
        #     cate_pred.append(torch.rand([1,size,size,80], device='cuda:0'))
        #
        seg_inputs = list(ori_seg_inputs)
        # avg = [-0.4037, -0.3930, -0.3284, -0.2290, -0.1640]
        # for i, size in enumerate([40, 36, 24, 16, 12]):
        #     seg_inputs[1][i] = torch.rand([1,256, size,size], device='cuda:0')*2*avg[i]*2

        # seg_inputs[2] += torch.rand(seg_inputs[2].size(), device='cuda:0')*2*seg_inputs[2].mean()
        seg_inputs[2] *= 2

        seg_result = self.bbox_head.get_seg(*seg_inputs)
        return seg_result

    def save_seg_result(self, seg_result, img_meta, save_dir='./seg_result/'):
        id = img_meta[0]['filename'].split('/')[-1].split('.')[0]
        save_dir = f'{save_dir}/{id}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i, tensor_mask in enumerate(seg_result[0][0]):
            tensor_mask = tensor_mask.cpu().type(torch.uint8)*0.8
            PIL_mask = torchvision.transforms.ToPILImage()(tensor_mask)
            PIL_mask.save(f'{save_dir}/{i}.png')
            # torchvision.transforms.ToPILImage()(torch.rand([400, 400]) * 255).save('./demo.png')
            # torchvision.transforms.ToPILImage()(torch.zeros([400, 400]) * 255).save('./demo.png')

    def vis_single_tensor(self, t, save_dir='./tmp/', name=0):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        t = t.cpu().type(torch.uint8) * 0.8
        PIL_mask = torchvision.transforms.ToPILImage()(t)
        PIL_mask.save(f'{save_dir}/vis_tensor_{name}.png')


    def vis_tensors(self, tensors, save_dir='./tmp/'):
        for i, t in enumerate(tensors):
            self.vis_single_tensor(t, name=i)




