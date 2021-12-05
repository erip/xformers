# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging

import pytest
import torch

_triton_available = torch.cuda.is_available()
if _triton_available:
    try:
        from xformers.triton import outer_product_mean
        from xformers.triton.utils import gpu_capabilities_older_than_70

    except ImportError:
        logging.warning(
            "Triton is not available, some optimizations will not be tested."
        )
        _triton_available = False

SHAPES = [(1, 128, 256), (1, 384, 128), (1, 784, 512)]


def reference_opm(a, b):
    # [*, N_res, N_res, C, C]
    outer = torch.einsum("...bac,...dae->...bdce", a, b)
    return outer


@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.skipif(
    not _triton_available or gpu_capabilities_older_than_70(),
    reason="Triton requires a SM70+ GPU",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_triton_outer_product_mean(shape, dtype):
    a = torch.rand(shape, dtype=dtype, device=torch.device("cuda"))
    b = torch.rand(shape, dtype=dtype, device=torch.device("cuda"))

    ref_opm = reference_opm(a, b)  # noqa
    triton_opm = outer_product_mean(
        a.transpose(-2, -1), b.transpose(-2, -1), average=False
    )  # noqa

    assert torch.allclose(ref_opm, triton_opm, rtol=0.01)
