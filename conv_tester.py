import hidet
import torch

from torch._inductor.triton_ops import conv as triton_conv
from hidet.ops import conv2d as hidet_conv
