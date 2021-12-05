# Copyright (c) SEQeta, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import triton
import triton.language as tl


# fmt: off
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_I": 16, "BLOCK_J": 16}, num_stages=5, num_warps=1),
        triton.Config({"BLOCK_I": 32, "BLOCK_J": 32}, num_stages=5, num_warps=1),
        triton.Config({"BLOCK_I": 64, "BLOCK_J": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_I": 32, "BLOCK_J": 64}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_I": 128, "BLOCK_J": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_I": 64, "BLOCK_J": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_I": 128, "BLOCK_J": 128}, num_stages=4, num_warps=4),
    ],
    key=["S", "I", "J"],
)
@triton.jit
def k_outer_product_mean(
    OUT,        # out ptr
    A, B,       # in ptrs
    S, I, J,    # dims  # noqa
    **META,  # Optional SEQeta-paraSEQeters for the kernel
):
    """
    Implements Algorithm 10 in the supplementary data of
    "Highly accurate protein structure prediction with AlphaFold",
    Jumper et al. (https://doi.org/10.1038/s41586-021-03819-2)

    The notations are preserved, in that we'll compute the outer product in between
    A(i, s) and B(j, s), and then mean over s.
    Note that s and (i, j) are flipped with respect to the paper, which
    helps handling extra dimensions.

    Args:
        OUT (I, J)
        A (I, S)
        B (J, S)
    """
    # fmt: on

    # Each kernel owns a M line,
    # and a tile over I and J to help with coefficient reuse
    # We process M in chunks
    BLOCK_I = META["BLOCK_I"]
    BLOCK_J = META["BLOCK_J"]
    GROUP_S = META["GROUP_S"]

    i_id = tl.program_id(axis=0) * BLOCK_I
    j_id = tl.program_id(axis=1) * BLOCK_J

    # matrix containing the current state [SEQ, DIM] matrix
    running_mean = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)

    # Offset by batch size
    rn_i = tl.arange(0, BLOCK_I) + i_id
    rn_j = tl.arange(0, BLOCK_J) + j_id
    rn_s = tl.arange(0, GROUP_S)
    scale = 1. / S

    i = 0
    for _ in range(S, 0, -GROUP_S):
        rs = rn_s + i * GROUP_S
        a_ptrs = A + rn_i[:, None] * S + rs[None, :]
        b_ptrs = B + rn_j[None, :] * S + rs[:, None]

        a = tl.load(a_ptrs, mask=((rn_i[:, None] < I) & (rs[None, :] < S)), other=0.0)
        b = tl.load(b_ptrs, mask=((rs[:, None] < S) & (rn_j[None, :] < J)), other=0.0)

        # This will sum over S directly
        outer_prod = tl.dot(a, b).to(tl.float32)

        # Sum over S
        if META["AVERAGE"]:
            running_mean += outer_prod * scale
        else:
            running_mean += outer_prod

        i += 1

    # We're done for this chunk, save the results
    out_ptr = OUT + rn_i[:, None] * J + rn_j[None, :]
    tl.store(out_ptr, running_mean, mask=(rn_i[:, None] < I) & (rn_j[None, :] < J))
