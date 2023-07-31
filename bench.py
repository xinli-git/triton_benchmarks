from pathlib import Path
import pandas as pd

import torch
import hidet
import logging

from torch import _dynamo

class Bench:

    def __init__(self):

        self.warmup_iters = 10
        self.bench_iters = 200
        self.bench_repeats = 5

        self.warn_variance = 0.1

        torch.backends.cudnn.benchmark = True

    @torch.no_grad()
    def bench(self, functor, args):

        for _ in range(self.warmup_iters):
            functor(*args)

        latencies = []
        for _ in range(self.bench_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(self.bench_iters):
                functor(*args)
            end.record()
            end.synchronize()
            latencies.append(start.elapsed_time(end) / self.bench_iters)

        mean = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)

        if (max_lat - min_lat) / mean > self.warn_variance:
            logging.warn("{} has high latency variance {}".format(self, (max_lat - min_lat) / mean))

        return mean, min_lat, max_lat, latencies

    @property
    def output_file(self):
        raise NotImplementedError

    @property
    def implementations(self):
        raise NotImplementedError

    @property
    def attrs(self):
        return NotImplementedError

    def data(self):
        raise NotImplementedError

    @staticmethod
    def in_df(attr, df):

        for k, v in attr.items():
            cond = (df[k] == v)
            if not cond.any():
                return False
            df = (df[cond])

        return len(df) == 3

    def run(self, root_dir):

        hidet.option.cache_dir(root_dir)
        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
        output = root_dir / self.output_file
        if output.exists():
            df = pd.read_csv(output)
            if self.in_df(self.attrs, df):
                print("Skipped {}, already exists".format(self))
                return
        else:
            df = pd.DataFrame(columns=[*self.attrs.keys(), 'impl', 'latency', 'min', 'max'])

        data = self.data()
        for name, impl in self.implementations:
            assert callable(impl)
            torch._dynamo.reset()
            latency, minn, maxx, _ = self.bench(impl, data)

            entry = self.attrs.copy()
            entry['impl'] = name
            entry['latency'] = latency
            entry['min'] = minn
            entry['max'] = maxx
            new_df = pd.DataFrame([entry])
            df = pd.concat([df, new_df], ignore_index=True)

        df.to_csv(output, index=False)
        return output.resolve()

