import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import ParameterTuple, Parameter
import mindspore.ops.functional as F
from mindspore.ops import ExpandDims
from mindspore.ops import composite as C
import mindspore.ops.operations as P


class TrainOneStepEndCell(nn.Cell):
    def __init__(self,net,optimizer,sens=1.0):
        super(TrainOneStepEndCell,self).__init__(auto_prefix=False)

        self.net=net
        
        # 使用tuple包装weight
        self.weights=ParameterTuple(self.net.trainable_params())
        
        
        # 使用优化器
        self.optimizer=optimizer
        # 定义梯度函数
        self.grad=C.GradOperation(get_by_list=True,sens_param=True)
        self.sens=sens
        
    def construct(self,image,img_orig,gt_uv_map,gt_keypoints_2d,gt_keypoints_3d,has_pose_3d,gt_keypoints_2d_smpl,gt_keypoints_3d_smpl,has_pose_3d_smpl,gt_pose,gt_betas,has_smpl,has_dp,gender,gt_dp_iuv,batch_size,ada_weight,step_count):
        #print(image.shape)
        #print(img_orig.shape)
        #print(gt_uv_map.shape)
        #print(gt_keypoints_2d.shape)
        #print(gt_keypoints_3d.shape)
        #print(has_pose_3d.shape)
        #print(gt_keypoints_2d_smpl.shape)
        #print(gt_keypoints_3d_smpl.shape)
        #print(has_pose_3d_smpl.shape)
        #print(gt_pose.shape,gt_betas.shape)
        #print(has_smpl.shape,has_dp.shape,gender.shape,gt_dp_iuv.shape,batch_size,ada_weight,step_count)
        #print("1")
        weights=self.weights
        loss=self.net(image,img_orig,gt_uv_map,gt_keypoints_2d,gt_keypoints_3d,has_pose_3d,gt_keypoints_2d_smpl,gt_keypoints_3d_smpl,has_pose_3d_smpl,gt_pose,gt_betas,has_smpl,has_dp,gender,gt_dp_iuv,batch_size,ada_weight,step_count)
        
        # 为反向传播设定系数
        #生成一个与loss的shape一样的张量，并且内容是1.0
        sens=P.Fill()(P.DType()(loss),P.Shape()(loss),self.sens)
        
        if type(ada_weight)!=Tensor:
            ada_weight=Tensor(1.0,mindspore.float32)
        batch_size=Tensor(batch_size,mindspore.int32)

        grad=self.grad(self.net,weights)
        grad=grad(image,img_orig,gt_uv_map,gt_keypoints_2d,gt_keypoints_3d,has_pose_3d,gt_keypoints_2d_smpl,gt_keypoints_3d_smpl,has_pose_3d_smpl,gt_pose,gt_betas,has_smpl,has_dp,gender,gt_dp_iuv,batch_size,ada_weight,sens=sens)
        loss=F.depend(loss,self.optimizer(grad))

        return loss

