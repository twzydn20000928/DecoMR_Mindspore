import mindspore
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore import Tensor
import numpy as np
from mindspore import ops


class ResizeBilinear(nn.Cell):

    def __init__(self, size=None, scale_factor=None, align_corners=False):
        """Initialize ResizeBilinear."""
        super(ResizeBilinear, self).__init__()
        self.name = type(self).__name__
        self.size = size
        self.scale_factor = float(scale_factor) if scale_factor else None
        self.align_corners = align_corners

    def construct(self, x):
        resize_bilinear = nn.ResizeBilinear()
        result = resize_bilinear(x, size=self.size, scale_factor=self.scale_factor,
                                 align_corners=self.align_corners)
        return result
