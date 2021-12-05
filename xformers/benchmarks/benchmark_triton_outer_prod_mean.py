# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict

import torch
import triton

from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.triton import outer_product_mean

SHAPES = [
    (1, 256, 256),
    (1, 512, 512),
    (1, 1024, 1024),
    (1, 2048, 2048),
    (1, 4096, 4096),
]


def to_gbs_fw(a, b, ms):
    # Read the two arrays, write the consolidated version
    return (
        (a.numel() + b.numel() + a.shape[-1] * b.shape[-1]) * a.element_size() * 1e-9
    ) / (ms * 1e-3)


def bench_outer_product_mean(avg):
    device = torch.device("cuda")

    for dtype in [
        torch.float16,
        torch.float32,
    ]:
        results: Dict[str, Any] = {}

        for B, M, K in SHAPES:
            a = torch.rand(B, M, K, device=device, dtype=dtype, requires_grad=False)
            b = torch.rand(B, M, K, device=device, dtype=dtype, requires_grad=False)

            def torch_step(x, y):
                z = torch.einsum("...bac,...dae->...bdce", x, y)
                if avg:
                    return z / x.shape[-2]
                return z

            def triton_step(x, y):
                return outer_product_mean(x, y, average=avg)

            for testcase in [
                TestCase(
                    torch_step,
                    "pytorch - avg{}".format(avg),
                ),
                TestCase(
                    triton_step,
                    "triton - avg{}".format(avg),
                ),
            ]:
                time = triton.testing.do_bench(lambda: testcase.function(a, b))[0]
                key = f"B={B}, M={M}, K={K}"
                if key not in results:
                    results[key] = {}

                # Record BW
                bandwidth = to_gbs_fw(a, b, time)
                results[key][testcase.name] = f"{bandwidth:.1f}"

        pretty_print(results, title="\n --- Type: {} --- ".format(dtype), units="GB/s")
        pretty_plot(
            results,
            title="OuterProduct-AVG{}-{}".format(avg, dtype),
            units="GB/s",
            dash_key="pytorch",
        )


for avg in [False, True]:
    bench_outer_product_mean(avg)
