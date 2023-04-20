import hidet
import torch

from torch._inductor.kernel.conv import convolution as _triton_conv
from hidet.graph.ops import conv2d as _hidet_conv
from torch import convolution as _torch_conv

import itertools

from bench import Bench

class ConvBench(Bench):

    def __init__(self, inp_shape, data_shape, stride, dilation, groups, dtype='float16', device='cuda'):
        super().__init__()

        assert(len(inp_shape) == 4 and len(data_shape) == 4)

        self.inp_shape = inp_shape
        self.data_shape = data_shape

        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.device = device
        self.dtype = dtype


    @property
    def output_file(self):
        return  "conv_benchmark_results.csv"

    @property
    def implementations(self):
        def torch_conv(data, weight, stride, dilation, groups):
            return _torch_conv(data, weight, None, stride, 0, dilation, groups)

        def triton_conv(data, weight, stride, dilation, groups):
            return _triton_conv(data, weight, None, stride, (0, 0), dilation, transposed=False,
                    output_padding=(0, 0), groups=groups)

        def hidet_conv(data, weight, stride, dilation, groups):
            data = hidet.from_torch(data)
            weight = hidet.from_torch(weight)
            with hidet.graph.PassContext() as ctx:
                ctx.set_parallel_k(search=True)
                ctx.set_mma('mma')
                return _hidet_conv(data, weight, stride, dilation, groups)

        return torch_conv, triton_conv, hidet_conv


    def __str__(self):

        s = "conv2d({}, {}), stride={}, dilation={}, groups={}".format(
                self.inp_shape, self.data_shape, self.stride, self.dilation, self.groups)

        return s

    @property
    def attrs(self):
        return {
                "device" : self.device,
                "dtype" : self.dtype,
                "inp_shape" : self.inp_shape,
                "data_shape" : self.data_shape,
                "stride" : self.stride,
                "dilation" : self.dilation,
                "groups" : groups
             }

    def data(self):
        dtype = getattr(torch, self.dtype)
        inp = torch.empty(self.inp_shape, dtype=dtype, device=self.device)
        weight = torch.empty(self.data_shape, dtype=dtype, device=self.device)

        return inp, weight

if __name__ == "__main__":

    inputs = [(1, 3, 224, 224,)]
    filters = [(3, 64, 7, 7)]
    dilation = [(1, 1)]
    stride = [(2, 2), (1, 1)]
    groups = [1]

    for config in itertools.product(inputs, filters, dilation, stride, groups):
        print(config)
        bench = ConvBench(*config)
        bench.run("benchmarks")

