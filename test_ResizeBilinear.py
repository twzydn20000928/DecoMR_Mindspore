from mindspore import Tensor
import mindspore.nn as nn
import torch
import numpy as np
from models.ResizeBilinear import *

# In MindSpore, it is predetermined to use bilinear to resize the input image.
x = np.random.randn(1, 2, 3, 4).astype(np.float32)
#resize = nn.ResizeBilinear()
tensor = Tensor(x)
t = ResizeBilinear(size=(5, 5))
output2 = t(tensor)
#output = resize(tensor, size=(5, 5))
#print(output.shape)
print(output2.shape)
