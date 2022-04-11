import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import ops
from models.geometric_layers import orthographic_projection, rodrigues
from utils.renderer import Renderer, visualize_reconstruction, vis_mesh
import time
class WithLossEndCell(nn.Cell):
    def __init__(self, CNet, LNet, criterion, options):
        super(WithLossEndCell, self).__init__(auto_prefix=False)
        self.CNet = CNet
        self.LNet = LNet
        # self.dp_loss=criterion
        self.options = options
        self.losses = None
        self.data = None
        self.bt = criterion

    def get_losses(self):
        return self.losses

    def construct(self, images,img_orig, gt_uv_map, gt_keypoints_2d, gt_keypoints_3d, has_pose_3d,
                  gt_keypoints_2d_smpl, gt_keypoints_3d_smpl, has_pose_3d_smpl, gt_pose, gt_betas, has_smpl, has_dp,
                  gender, gt_dp_iuv, batch_size,ada_weight,step_count=None):

        if ada_weight!=None and ada_weight.ndim==0 and ada_weight.asnumpy()==1.0:
            ada_weight=None
        if type(batch_size)==Tensor:
            batch_size=int(batch_size.asnumpy())
        time1=time.time()
        dtype = mindspore.float32
        pred_dp, dp_feature, codes = self.CNet(images)
        pred_uv_map, pred_camera = self.LNet(pred_dp, dp_feature, codes)
        self.pred_dp = pred_dp
        self.pred_uv_map = pred_uv_map
        self.pred_camera = pred_camera
        time2=time.time()
        print("m",time2-time1)

        # if self.options.adaptive_weight:
        #     fit_joint_error = input_batch['fit_joint_error']
        #     ada_weight = self.error_adaptive_weight(fit_joint_error).type(dtype)
        # else:
        #     # ada_weight = pred_scale.new_ones(batch_size).type(dtype)
        #     ada_weight = None
        #ada_weight = None
        losses = {}
        '''loss on dense pose result'''
        time1=time.time()
        loss_dp_mask, loss_dp_uv = self.bt.dp_loss(pred_dp, gt_dp_iuv, has_dp, ada_weight)
        loss_dp_mask = loss_dp_mask * self.options.lam_dp_mask
        loss_dp_uv = loss_dp_uv * self.options.lam_dp_uv
        losses['dp_mask'] = loss_dp_mask
        losses['dp_uv'] = loss_dp_uv
        

        '''loss on location map'''
        
        sampled_vertices = self.bt.sampler.resample(pred_uv_map.astype(dtype)).astype(dtype)
        #loss_uv = self.bt.uv_loss(gt_uv_map.astype(dtype), pred_uv_map.astype(dtype), has_smpl, ada_weight).astype(
        #    dtype) * self.options.lam_uv
        #losses['uv'] = loss_uv
        

        if self.options.lam_tv > 0:
            loss_tv = self.bt.tv_loss(pred_uv_map) * self.options.lam_tv
            losses['tv'] = loss_tv

        '''loss on mesh'''
        if self.options.lam_mesh > 0:
            loss_mesh = self.bt.shape_loss(sampled_vertices, gt_vertices, has_smpl, ada_weight) * self.options.lam_mesh
            losses['mesh'] = loss_mesh

        '''loss on joints'''
        weight_key = ops.Ones()(batch_size,sampled_vertices.dtype)
        if self.options.gtkey3d_from_mesh:
            # For the data without GT 3D keypoints but with SMPL parameters,
            # we can get the GT 3D keypoints from the mesh.gt_vertices
            # The confidence of the keypoints is related to the confidence of the mesh.
            gt_keypoints_3d_mesh = self.bt.smpl.get_train_joints()
            gt_keypoints_3d_mesh = ops.Concat(-1)([gt_keypoints_3d_mesh, ops.Ones()([batch_size, 24, 1],gt_keypoints_3d_mesh.dtype)])
            valid = has_smpl > has_pose_3d

            gt_keypoints_3d=gt_keypoints_3d.asnumpy()
            gt_keypoints_3d[valid.asnumpy()] = gt_keypoints_3d_mesh.asnumpy()[valid.asnumpy()]   
            
            
            
            gt_keypoints_3d=Tensor(gt_keypoints_3d)
            
            has_pose_3d=has_pose_3d.asnumpy()
            has_pose_3d[valid.asnumpy()] = 1
            has_pose_3d=Tensor(has_pose_3d)

            if ada_weight is not None:
                weight_key=weight_key.asnumpy()
                weight_key[valid.asnumpy()] = ada_weight.asnumpy()[valid.asnumpy()]   
                weight_key=Tensor(weight_key)

        sampled_joints_3d = self.bt.smpl.get_train_joints(sampled_vertices)
        
        loss_keypoints_3d = self.bt.keypoint_3d_loss(sampled_joints_3d, gt_keypoints_3d, has_pose_3d, weight_key)
        loss_keypoints_3d = loss_keypoints_3d * self.options.lam_key3d
        losses['key3D'] = loss_keypoints_3d
        
        
        sampled_joints_2d = orthographic_projection(sampled_joints_3d, pred_camera)[:, :, :2]
        loss_keypoints_2d = self.bt.keypoint_loss(sampled_joints_2d, gt_keypoints_2d) * self.options.lam_key2d
        losses['key2D'] = loss_keypoints_2d
        

        # We add the 24 joints of SMPL model for the training on SURREAL dataset.
        weight_key_smpl = ops.Ones()(batch_size,sampled_vertices.dtype)
        if self.options.gtkey3d_from_mesh:
            gt_keypoints_3d_mesh = self.bt.smpl.get_smpl_joints(gt_vertices)
            gt_keypoints_3d_mesh = ops.Concat(-1)([gt_keypoints_3d_mesh,ops.Ones()([batch_size, 24, 1],gt_keypoints_3d_mesh.dtype)])
            valid = has_smpl > has_pose_3d_smpl
            gt_keypoints_3d_smpl=gt_keypoints_3d_smpl.asnumpy()
            gt_keypoints_3d_smpl[valid.asnumpy()] = gt_keypoints_3d_mesh.asnumpy()[valid.asnumpy()]

            has_pose_3d_smpl=has_pose_3d_smpl.asnumpy()
            has_pose_3d_smpl[valid.asnumpy()] = 1
            has_pose_3d_smpl=Tensor(has_pose_3d_smpl)
            if ada_weight is not None:
                weight_key_smpl=weight_key_smpl.asnumpy()
                weight_key_smpl[valid.asnumpy()] = ada_weight.asnumpy()[valid.asnumpy()]
                weight_key_smpl=Tensor(weight_key_smpl)

        if self.options.use_smpl_joints:
            sampled_joints_3d_smpl = self.bt.smpl.get_smpl_joints(sampled_vertices)
            loss_keypoints_3d_smpl = self.bt.smpl_keypoint_3d_loss(sampled_joints_3d_smpl, gt_keypoints_3d_smpl,
                                                                       has_pose_3d_smpl, weight_key_smpl)
            loss_keypoints_3d_smpl = loss_keypoints_3d_smpl * self.options.lam_key3d_smpl
            losses['key3D_smpl'] = loss_keypoints_3d_smpl

            sampled_joints_2d_smpl = orthographic_projection(sampled_joints_3d_smpl, pred_camera)[:, :, :2]
            loss_keypoints_2d_smpl = self.bt.keypoint_loss(sampled_joints_2d_smpl,
                                                               gt_keypoints_2d_smpl) * self.options.lam_key2d_smpl
            losses['key2D_smpl'] = loss_keypoints_2d_smpl

        '''consistent loss'''
        
        if not self.options.lam_con == 0:
            loss_con = self.bt.consistent_loss(gt_dp_iuv, pred_uv_map, pred_camera,ada_weight) * self.options.lam_con
            losses['con'] = loss_con
        
        
        self.losses = losses
        loss_total = sum(loss for loss in losses.values())

        if step_count!=None and step_count!=None and (step_count + 1) % self.options.summary_steps == 0:
            data = {}
            vis_num = min(4, batch_size)
            data['image'] = img_orig[0:vis_num].copy()
            data['gt_vert'] = gt_vertices[0:vis_num].copy()
            data['pred_vert'] = sampled_vertices[0:vis_num].copy()
            data['pred_cam'] = pred_camera[0:vis_num].copy()
            data['pred_joint'] = sampled_joints_2d[0:vis_num].copy()
            data['gt_joint'] = gt_keypoints_2d[0:vis_num].copy()
            data['pred_uv'] = pred_uv_map[0:vis_num].copy()
            data['gt_uv'] = gt_uv_map[0:vis_num].copy()
            data['pred_dp'] = pred_dp[0:vis_num].copy()
            data['gt_dp'] = gt_dp_iuv[0:vis_num].copy()
            self.data = data
        #print("loss",loss_total)
        time2=time.time()
        print("loss",time2-time1)
        print("------------------------------------")
        return loss_total

