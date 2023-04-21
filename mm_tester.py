import hidet
import torch

from torch._inductor.kernel.conv import convolution as _triton_conv
from hidet.graph.ops import conv2d as _hidet_conv
from torch import convolution as _torch_conv

import itertools

from bench import Bench

class ConvBench(Bench):

    def __init__(self, inp_shape, data_shape, stride, dilation, groups, dtype='float32', device='cuda'):
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
            return torch.conv2d(data, weight, None, stride, (0, 0), dilation, groups=groups)

        triton_conv = torch.compile(torch_conv, mode='max-autotune')

        # hidet.torch.dynamo_config.search_space(2)
        # hidet.torch.dynamo_config.use_tensor_core()
        # hidet.torch.dynamo_config.parallel_k('search')
        # hidet.torch.dynamo_config.use_cuda_graph()
        # hidet.torch.dynamo_config.dump_graph_ir("hidet_graph")
        # hidet_conv = torch.compile(torch_conv, backend='hidet')
        # def triton_conv(data, weight, stride, dilation, groups):
        #     return _torch_conv(data, weight, None, stride, (0, 0), dilation, transposed=False,
        #             output_padding=(0, 0), groups=groups)

        data, weight, stride, dilation, groups = self.data()
        data = hidet.from_torch(data)
        weight = hidet.from_torch(weight)

        symbol_data = hidet.symbol_like(data)
        symbol_weight = hidet.symbol_like(weight)
        symbol_output = hidet.ops.conv2d(symbol_data, symbol_weight, stride, dilation, groups)
        graph: hidet.FlowGraph = hidet.trace_from(symbol_output, inputs=[symbol_data, symbol_weight])

        hidet.option.search_space(2)
        with hidet.graph.PassContext() as ctx:
            ctx.set_parallel_k(disabled=True)
            ctx.set_mma('mma')
            graph_opt: hidet.FlowGraph = hidet.graph.optimize(graph)
            cuda_graph = graph_opt.cuda_graph()
            print(graph_opt)

        def hidet_conv(data, weight, stride, dilation, groups):
            return cuda_graph.run()
            data = hidet.from_torch(data)
            weight = hidet.from_torch(weight)
            return cuda_graph.run([data, weight])

        return [("torch", torch_conv), ("triton", triton_conv), ("hidet", hidet_conv)]


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

        return inp, weight, self.stride, self.dilation, self.groups

if __name__ == "__main__":
    import sys
    special_config = (
            [(1, 3, 224, 224,), (64, 3, 7, 7)],
            (2, 2),
            (1, 1),
            1,
    )
    inp_filters = [
            [(1, 64, 64, 64,), (128, 64, 3, 3)],
            [(1, 128, 32, 32,), (256, 128, 1, 1)],
            [(1, 256, 16, 16,), (512, 256, 3, 3)],
            [(1, 512, 8, 8,), (2048, 512, 1, 1)],
            ]
    stride = [(2, 2), (1, 1)]
    dilation = [(1, 1)]
    groups = [1, 4]
    dtype = ['float16', 'float32']
    sweep_config = list(itertools.product(inp_filters, stride, dilation, groups, dtype))

    sweep_config.append(special_config + ('float16',))
    sweep_config.append(special_config + ('float32',))
    for config in sweep_config:
        (inp, fil), _stride, _dilation, _groups, _dtype = config
        if _groups > 1:
            fil = list(fil)
            fil[1] = int(fil[1] /_groups)
        bench = ConvBench(inp, fil, _stride, _dilation, _groups, _dtype)
        print("Testing:\n{}".format(bench))
        bench.run(sys.argv[1])

