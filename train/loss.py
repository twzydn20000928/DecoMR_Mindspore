
from __future__ import division
import sys
import time
#sys.path.append("../")
import mindspore
import mindspore.nn as nn
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
# from models.grid_sample import GridSample
from mindspore import load_checkpoint,load_param_into_net

class Loss(object):
    """
    Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """
    def __init__(self, options):
        self.options = options
        self.endtime = time.time() + self.options.time_to_run

        # override this function to define your model, optimizers etc.
        
        # Create loss functions
        self.criterion_shape = nn.L1Loss()#.to(self.device)
        self.criterion_uv = nn.L1Loss()#.to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none')#.to(self.device)
        self.criterion_keypoints_3d = nn.L1Loss(reduction='none')#.to(self.device)
        self.criterion_regr = nn.MSELoss()#.to(self.device)
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
            '''
            if weight is not None:
                weight = weight[has_dp > 0, None, None, None]
            else:
                weight = 1.0
            '''
            if weight!=1.0:
                weight = weight[has_dp > 0, None, None, None]
            weight = Tensor(weight, mindspore.float32)

            min_value = Tensor(0.0, mindspore.float32)
            max_value = Tensor(1.0, mindspore.float32)
            # print("pred_mask_shape.shape = ", pred_mask_shape)

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
    
    
    def error_adaptive_weight(self, fit_joint_error):
        weight = (1 - 10 * fit_joint_error)
        weight[weight <= 0] = 0
        return weight

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, weight=None):
        """
        Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the weight
        The available keypoints are different for each dataset.
        """
        if gt_keypoints_2d.shape[2] == 3:
            conf=ops.ExpandDims()(gt_keypoints_2d[:, :, -1],-1).copy()
        else:
            conf = 1

        if weight is not None:
            weight = weight[:, None, None]
            conf = conf * weight

        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, weight=None):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the weight
        """
        if gt_keypoints_3d.shape[2] == 3:
            tmp=ops.Ones()((gt_keypoints_3d.shape[0], gt_keypoints_3d.shape[1], 1),gt_keypoints_3d.dtype)
            gt_keypoints_3d = ops.Concat(2)((gt_keypoints_3d, tmp))

        conf = ops.ExpandDims()(gt_keypoints_3d[:, :, -1],-1).copy()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].copy()
        gt_keypoints_3d = Tensor(gt_keypoints_3d.asnumpy()[has_pose_3d.asnumpy() == 1])
        conf = Tensor(conf.asnumpy()[has_pose_3d.asnumpy() == 1])

        if weight is not None:
            weight = Tensor(weight.asnumpy()[has_pose_3d.asnumpy() == 1, None, None])
            conf = conf * weight

        pred_keypoints_3d = Tensor(pred_keypoints_3d.asnumpy()[has_pose_3d.asnumpy() == 1])
        if len(gt_keypoints_3d) > 0:
            # Align the origin of the first 24 keypoints with the pelvis.
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]

            # # Align the origin of the first 24 keypoints with the pelvis.
            # gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            # pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            # gt_keypoints_3d[:, :24, :] = gt_keypoints_3d[:, :24, :] - gt_pelvis[:, None, :]
            # pred_keypoints_3d[:, :24, :] = pred_keypoints_3d[:, :24, :] - pred_pelvis[:, None, :]
            #
            # # Align the origin of the 24 SMPL keypoints with the root joint.
            # gt_root_joint = gt_keypoints_3d[:, 24]
            # pred_root_joint = pred_keypoints_3d[:, 24]
            # gt_keypoints_3d[:, 24:, :] = gt_keypoints_3d[:, 24:, :] - gt_root_joint[:, None, :]
            # pred_keypoints_3d[:, 24:, :] = pred_keypoints_3d[:, 24:, :] - pred_root_joint[:, None, :]

            return (conf * self.criterion_keypoints_3d(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return ops.Fill()(mindspore.float32,(1,),0.)#.to(self.device)

    def smpl_keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, weight=None):
        """
        Compute 3D SMPL keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the weight
        """
        if gt_keypoints_3d.shape[2] == 3:
            
            tmp=ops.Ones()((gt_keypoints_3d.shape[0], gt_keypoints_3d.shape[1], 1),gt_keypoints_3d.dtype)
            gt_keypoints_3d = ops.Concat(2)((gt_keypoints_3d, tmp))
            

        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).copy()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].copy()
        gt_keypoints_3d = Tensor(gt_keypoints_3d.asnumpy()[has_pose_3d.asnumpy() == 1])
        conf = Tensor(conf.asnumpy()[has_pose_3d.asnumpy() == 1])

        if weight is not None:
            weight = Tensor(weight.asnumpy()[has_pose_3d.asnumpy() == 1, None, None])
            conf = conf * weight

        pred_keypoints_3d = Tensor(pred_keypoints_3d.asnumpy()[has_pose_3d.asnumpy() == 1])
        if len(gt_keypoints_3d) > 0:
            gt_root_joint = gt_keypoints_3d[:, 0, :]
            pred_root_joint = pred_keypoints_3d[:, 0, :]
            gt_keypoints_3d = gt_keypoints_3d - gt_root_joint[:, None, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_root_joint[:, None, :]
            return (conf * self.criterion_keypoints_3d(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return ops.Fill()(mindspore.float32,(1,),0.)#.to(self.device)
        
    def shape_loss(self, pred_vertices, gt_vertices, has_smpl, weight=None):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = Tensor(pred_vertices.asnumpy()[has_smpl.asnumpy() == 1])
        gt_vertices_with_shape = Tensor(gt_vertices.asnumpy()[has_smpl.asnumpy() == 1])

        if weight is not None:
            weight = Tensor(weight.asnumpy()[has_smpl.asnumpy() == 1, None, None])
        else:
            weight = 1

        if len(gt_vertices_with_shape) > 0:
            loss = self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
            loss = (loss * weight).mean()
            return loss
        else:
            return ops.Fill(mindspore.float32,(1,),0.)#.to(self.device)

    def uv_loss(self, pred_uv_map, gt_uv_map, has_smpl, weight=None):
        # self.uv_mask = self.uv_mask.to(pred_uv_map.device)
        self.uv_weight = self.uv_weight.astype(pred_uv_map.dtype)#.to(pred_uv_map.device)
        max = self.uv_weight.max()
        pred_uv_map_shape = Tensor(pred_uv_map.asnumpy()[has_smpl.asnumpy() == 1])
        gt_uv_map_with_shape = Tensor(gt_uv_map.asnumpy()[has_smpl.asnumpy() == 1])
        if len(gt_uv_map_with_shape) > 0:
            # return self.criterion_uv(pred_uv_map_shape * self.uv_mask, gt_uv_map_with_shape * self.uv_mask)
            if weight is not None:
                ada_weight = Tensor(weight.asnumpy()[has_smpl.asnumpy() > 0, None, None, None])
            else:
                ada_weight = 1.0
            loss = self.criterion_uv(pred_uv_map_shape * self.uv_weight, gt_uv_map_with_shape * self.uv_weight)
            loss = (loss * ada_weight).mean()
            return loss

        else:
            # return torch.FloatTensor(1).fill_(0.).to(self.device)
            return Tensor(0.0, dtype=pred_uv_map.dtype)#, device=self.device

    def tv_loss(self, uv_map):
        #self.uv_weight = self.uv_weight.to(uv_map.device)
        tv = ops.Abs()(uv_map[:,0:-1, 0:-1, :] - uv_map[:,0:-1, 1:, :]) \
            + ops.Abs()(uv_map[:,0:-1, 0:-1, :] - uv_map[:,1:, 0:-1, :])
        return ops.ReduceSum()(tv) / self.tv_factor
        # return torch.sum(tv * self.uv_weight[:, 0:-1, 0:-1]) / self.tv_factor

    def consistent_loss(self, dp, uv_map, camera, weight=None):

        tmp=npm.arange(0,dp.shape[-1], 1, dtype=dp.dtype)/(dp.shape[-1] -1)#, device=dp.device
        tmp = tmp * 2 - 1
        loc_y,loc_x = ops.Meshgrid(indexing="ij")((tmp,tmp))
    
        loc = ops.Stack(0)((loc_x, loc_y)).repeat(dp.shape[0],0)
        dp_mask = ops.ExpandDims()((dp[:, 0] > 0.5).astype(mindspore.float32),1)
        loc = dp_mask * loc

        dp_tmp = dp_mask * (dp[:, 1:] * 2 - 1)
        '''uv_map need to be transfered to img coordinate first'''
        uv_map = uv_map[:, :, :, :-1]
        camera = camera.view(-1, 1, 1, 3)
        uv_map = uv_map + camera[:, :, :, 1:]       # trans
        uv_map = uv_map * ops.ExpandDims()(camera[:, :, :, 0],-1)        # scale
        warp_loc = GridSample(uv_map.transpose(0, 3, 1, 2), dp_tmp.transpose(0, 2, 3, 1))[:, :2]
        warp_loc = warp_loc * dp_mask

        if weight is not None:
            weight = weight[:, None, None, None]
            dp_mask = dp_mask * weight

        loss_con = nn.MSELoss()(warp_loc * dp_mask, loc * dp_mask)
        return loss_con


        
