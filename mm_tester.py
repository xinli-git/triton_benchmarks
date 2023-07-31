import hidet
import torch
import triton

import itertools

from bench import Bench

class GemmBench(Bench):

    def __init__(self, M, N, K, dtype='float32', device='cuda'):
        super().__init__()

        self.M = M
        self.N = N
        self.K = K
        self.device = device
        self.dtype = dtype


    @property
    def output_file(self):
        return  "gemm_benchmark_results.csv"

    @property
    def implementations(self):
        def torch_gemm(matrixA, matrixB):
            return torch.matmul(matrixA, matrixB)

        def triton_gemm(matrixA, matrixB):
            return triton.ops.matmul(matrixA, matrixB)



        # hidet.torch.dynamo_config.search_space(2)
        # hidet.torch.dynamo_config.use_tensor_core()
        # hidet.torch.dynamo_config.parallel_k('search')
        # hidet.torch.dynamo_config.use_cuda_graph()
        # hidet.torch.dynamo_config.dump_graph_ir("hidet_graph")
        # hidet_conv = torch.compile(torch_conv, backend='hidet')
        # def triton_conv(data, weight, stride, dilation, groups):
        #     return _torch_conv(data, weight, None, stride, (0, 0), dilation, transposed=False,
        #             output_padding=(0, 0), groups=groups)

        a, b = self.data()
        a = hidet.from_torch(a)
        b = hidet.from_torch(b)

        symbol_a = hidet.symbol_like(a)
        symbol_b = hidet.symbol_like(b)
        symbol_output = hidet.ops.matmul(symbol_a, symbol_b)
        graph: hidet.FlowGraph = hidet.trace_from(symbol_output, inputs=[symbol_a, symbol_b])

        hidet.option.search_space(2)
        with hidet.graph.PassContext() as ctx:
            ctx.set_parallel_k(disabled=True)
            ctx.set_mma('mma')
            graph_opt: hidet.FlowGraph = hidet.graph.optimize(graph)
            cuda_graph = graph_opt.cuda_graph()

        def hidet_gemm(inp_a, inp_b):
            return cuda_graph.run()

        return [("torch", torch_gemm), ("triton", triton_gemm), ("hidet", hidet_gemm)]


    def __str__(self):

        s = "gemm, m={}, n={}, k={}".format(
                self.M, self.N, self.K)

        return s

    @property
    def attrs(self):
        return {
                "device" : self.device,
                "dtype" : self.dtype,
                "M" : self.M,
                "N" : self.N,
                "K" : self.K,
             }

    def data(self):
        dtype = getattr(torch, self.dtype)
        a = torch.empty((self.M, self.K), dtype=dtype, device=self.device)
        b = torch.empty((self.K, self.N), dtype=dtype, device=self.device)

        return a, b

if __name__ == "__main__":
    import sys
    Ms = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    Ns = [8, 16, 32, 64, 128, 256, 512]
    Ks = [32, 64, 128, 256, 512, 1024]
    dtype = ['float16', 'float32']
    sweep_config = list(itertools.product(Ms, Ns, Ks, dtype))

    for arg in sweep_config:
        bench = GemmBench(*arg)
        print("Testing:\n{}".format(bench))
        bench.run(sys.argv[1])

