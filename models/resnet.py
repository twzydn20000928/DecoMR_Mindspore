"""
This file contains the definitions of the various ResNet models.
Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py.
Forward pass was modified to discard the last fully connected layer
"""
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import export, load_checkpoint, load_param_into_net
import mindspore.common as common
from mindspore import Tensor,Parameter
from models.Initializer import *
from mindspore.common.initializer import HeNormal,Constant

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Bottleneck(nn.Cell):
    expansion=4

    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self,x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #print("resnetin")
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        #print("resnetout")
        return out


class ResNetBackbone(nn.Cell):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Dense(512 * block.expansion, num_classes)

        for n, m in self.cells_and_names():

            if isinstance(m, nn.Conv2d):

                c_weight = m.weight.asnumpy()
                HeNormal(mode="fan_in", nonlinearity="relu")(c_weight)
                m.weight.set_data(Tensor(c_weight))

            elif isinstance(m, nn.BatchNorm2d):

                b_gamma = m.gamma.asnumpy()
                Constant(1)(b_gamma)
                m.gamma.set_data(Tensor(b_gamma))

                b_beta = m.beta.asnumpy()
                Constant(0)(b_beta)
                m.beta.set_data(Tensor(b_beta))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)  # (, 64, 112, 112)
        
        x2 = self.maxpool(x1)
        
        x2 = self.layer1(x2)  # (, 256, 56, 56)
        
        x3 = self.layer2(x2)  # (, 512, 28, 28)

        x4 = self.layer3(x3)  # (, 1024, 14, 14)
 
        x5 = self.layer4(x4)  # (, 2048, 7, 7)
      
        y = self.avgpool(x5)
      
        y = y.view(y.shape[0], -1)  # (, 2048)
        feature_list = []
        feature_list.append(x)
        # feature_list.append(None)
        feature_list.append(x1)
        feature_list.append(x2)
        feature_list.append(x3)
        feature_list.append(x4)
        feature_list.append(x5)
        
        return y, tuple(feature_list)


def resnet50backbone(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetBackbone(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        ckpt_file_name = "models/ms_resnet50.ckpt"
        param_dict = load_checkpoint(ckpt_file_name)

        not_load_param = load_param_into_net(model, param_dict, strict_load=True)
        # load_param_into_net(model, load_checkpoint(model_urls["resnet50"]))
        if not_load_param:
            raise ValueError("Load param into network fail!")
    return model
