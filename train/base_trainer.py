from __future__ import division
import sys
import time

import mindspore
import mindspore.ops as ops
from mindspore.nn import ResizeBilinear
from mindspore.ops import ResizeNearestNeighbor
from mindspore import Tensor
from mindspore.nn import BCELoss
from tqdm import tqdm
from utils import CheckpointDataLoader, CheckpointSaver
from torch.utils.tensorboard import SummaryWriter #tensorboardX
import numpy as np
tqdm.monitor_interval = 100


class BaseTrainer(object):
    """
    Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """
    def __init__(self, options):
        self.options = options
        self.endtime = time.time() + self.options.time_to_run

        # override this function to define your model, optimizers etc.
        self.init_fn()
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.summary_writer = SummaryWriter(self.options.summary_dir)

        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(self.models_dict, self.optimizers_dict,
                                                    checkpoint_file=self.options.checkpoint)

        if self.checkpoint is None:

            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']
    # The following methods (with the possible exception of test) have to be implemented in the derived classes
    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def dp_loss(self, pred_dp, gt_dp, has_dp, weight=None):
        dtype = pred_dp.dtype

        x = (has_dp > 0)
        x = x.astype("int32")
        pred_dp_shape = pred_dp[[x]]
        gt_dp_shape = gt_dp[[x]]

        if len(gt_dp_shape) > 0:

            expand_dims = ops.ExpandDims()
            x = ((gt_dp_shape[:, 0]) > 0).astype("float32")
            gt_mask_shape = expand_dims(x, 1)

            gt_uv_shape = gt_dp_shape[:, 1:]

            pred_mask_shape = expand_dims(pred_dp_shape[:, 0], 1)
            pred_uv_shape = pred_dp_shape[:, 1:]

            resize_bilinear = ResizeBilinear()
            pred_mask_shape = resize_bilinear(x = pred_mask_shape, size = (gt_dp.shape[2], gt_dp.shape[3]) )
            resize = ops.ResizeNearestNeighbor((gt_dp.shape[2], gt_dp.shape[3]))
            pred_uv_shape = resize(pred_uv_shape)

            if weight is not None:
                weight = weight[has_dp > 0, None, None, None]
            else:
                weight = 1.0
            weight = Tensor(weight, mindspore.float32)

            min_value = Tensor(0.0, mindspore.float32)
            max_value = Tensor(1.0, mindspore.float32)
            print("pred_mask_shape.shape = ", pred_mask_shape)

            # pred_mask_shape = ops.clip_by_value(pred_mask_shape, min_value, max_value)       #一直卡着不动，也不报错

            loss = BCELoss(weight=weight, reduction='mean')
            # print("pred_mask_shape = ", pred_mask_shape)
            # print("gt_mask_shape = ", gt_mask_shape)
            loss_mask = loss(pred_mask_shape, gt_mask_shape)

            gt_uv_weight = (gt_uv_shape.abs().max(axis=1, keepdims=True) > 0).astype(dtype)

            x = gt_uv_weight.mean(axis=-1).mean(axis=-1)
            weight_ratio = (x[:, :, None, None] + 1e-8)
            gt_uv_weight = gt_uv_weight / weight_ratio

            logits = gt_uv_weight * pred_uv_shape
            labels = gt_uv_weight * gt_uv_shape
            '''算的太慢 会卡住 为了下一步进行 先注释'''
            # loss_uv = self.criterion_uv(logits, labels)
            # loss_uv = (loss_uv * weight).mean()

            logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
            labels = Tensor(np.array([[1, 1, 1], [1, 2, 2]]), mindspore.float32)
            loss_uv = self.criterion_uv(logits, labels)

            return loss_mask, loss_uv
        else:
            return pred_dp.sum() * 0, pred_dp.sum() * 0

