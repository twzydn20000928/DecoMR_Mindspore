import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import ParameterTuple, Parameter
import mindspore.ops.functional as F
from mindspore.ops import ExpandDims
from mindspore.ops import composite as C
import mindspore.ops.operations as P


class TrainOneStepCell(nn.Cell):
    def __init__(self, net, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)

        self.net = net

        # 使用tuple包装weight
        self.weights = ParameterTuple(self.net.trainable_params())
        # 使用优化器
        self.optimizer = optimizer
        # 定义梯度函数
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens=sens

    def construct(self, images, gt_dp_iuv, img_orig, has_dp, ada_weight, step_count):
        weights = self.weights
        #print("1")
        loss = self.net(images, gt_dp_iuv, has_dp, ada_weight, step_count)
        # 为反向传播设定系数
        # 生成一个与loss的shape一样的张量，并且内容是1.0
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        if type(ada_weight)!=Tensor:
            ada_weight=Tensor(1.0, mindspore.float32)
        
        grad = self.grad(self.net, weights)(images, gt_dp_iuv, img_orig, has_dp, ada_weight, Tensor(step_count), sens=sens)
        #print(grad[0,0])
        loss = F.depend(loss, self.optimizer(grad))
        #print("loss",loss)
        return loss

