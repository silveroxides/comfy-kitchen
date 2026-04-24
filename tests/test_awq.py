# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWQ W4A16 smoke tests.

Covers the eager reference path for kitchen's AWQ GEMV. The op is eager-only
(no CUDA backend yet), so these tests exercise the public API wrapper, the
torch.library dispatch, and the pack/unpack/dequant round-trip against a
fp32 reference.
"""
from __future__ import annotations

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.eager.awq import _unpack_uint4_row_major

from .conftest import assert_values_close


def _pack_uint4_row_major(unpacked: torch.Tensor) -> torch.Tensor:
    """(..., K) int8 in [0,15] -> (..., K//2) int8 with two uint4 per byte.

    Even column goes to the low nibble, odd column to the high nibble,
    matching _unpack_uint4_row_major.
    """
    x32 = unpacked.to(torch.int32) & 0x0F
    lo = x32[..., 0::2]
    hi = x32[..., 1::2]
    return (lo | (hi << 4)).to(torch.int8)


def _dequantize_reference(
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    wzeros: torch.Tensor,
    group_size: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """fp32 reference dequantization: W[n,k] = (q - 8) * s + z."""
    N, K_half = qweight.shape
    K = K_half * 2
    w_uint = _unpack_uint4_row_major(qweight).to(compute_dtype)
    w_groups = w_uint.view(N, K // group_size, group_size)
    scales_ng = wscales.t().unsqueeze(-1)
    zeros_ng = wzeros.t().unsqueeze(-1)
    return ((w_groups - 8.0) * scales_ng + zeros_ng).view(N, K)


class TestAwqUnpack:
    """Pack/unpack round-trip on the kitchen-native uint4 row-major layout."""

    def test_unpack_known_bytes(self):
        # 0xF0 -> lo=0, hi=15; 0x5A -> lo=10, hi=5
        packed = torch.tensor([[0xF0, 0x5A]], dtype=torch.int8)
        expected = torch.tensor([[0, 15, 10, 5]], dtype=torch.int8)
        assert torch.equal(_unpack_uint4_row_major(packed), expected)

    def test_pack_unpack_roundtrip(self, seed):
        N, K = 8, 128
        vals = torch.randint(0, 16, (N, K), dtype=torch.int8)
        packed = _pack_uint4_row_major(vals)
        unpacked = _unpack_uint4_row_major(packed)
        assert torch.equal(unpacked, vals)


class TestAwqForward:
    """End-to-end forward through the public API against a fp32 reference."""

    @pytest.mark.parametrize(
        "M,N,K,group_size",
        [
            (1, 32, 64, 64),
            (4, 64, 128, 64),
            (8, 128, 256, 64),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_forward_matches_reference(self, seed, M, N, K, group_size, dtype):
        K_half = K // 2
        num_groups = K // group_size

        qweight_vals = torch.randint(0, 16, (N, K), dtype=torch.int8)
        qweight = _pack_uint4_row_major(qweight_vals)

        wscales = torch.rand(num_groups, N, dtype=dtype) * 0.1 + 0.01
        wzeros = (torch.rand(num_groups, N, dtype=dtype) - 0.5) * 0.05
        x = torch.randn(M, K, dtype=dtype) * 0.3

        with ck.use_backend("eager"):
            out = ck.gemv_awq_w4a16(x, qweight, wscales, wzeros, group_size=group_size)

        w_ref = _dequantize_reference(qweight, wscales, wzeros, group_size, dtype)
        expected = x.to(dtype) @ w_ref.t()

        assert out.shape == (M, N)
        assert out.dtype == dtype
        if dtype == torch.float32:
            rtol, atol = 1e-5, 1e-5
        else:
            rtol, atol = 5e-3, 5e-3
        assert_values_close(out, expected, rtol=rtol, atol=atol, name="awq gemv vs fp32 reference")

    def test_with_bias(self, seed):
        M, N, K, group_size = 4, 32, 128, 64
        dtype = torch.bfloat16
        num_groups = K // group_size

        qweight_vals = torch.randint(0, 16, (N, K), dtype=torch.int8)
        qweight = _pack_uint4_row_major(qweight_vals)
        wscales = torch.rand(num_groups, N, dtype=dtype) * 0.1 + 0.01
        wzeros = torch.zeros(num_groups, N, dtype=dtype)
        x = torch.randn(M, K, dtype=dtype) * 0.3
        bias = torch.randn(N, dtype=dtype) * 0.5

        with ck.use_backend("eager"):
            out_with = ck.gemv_awq_w4a16(x, qweight, wscales, wzeros, bias=bias, group_size=group_size)
            out_without = ck.gemv_awq_w4a16(x, qweight, wscales, wzeros, group_size=group_size)

        diff = out_with - out_without
        expected_diff = bias.expand_as(diff)
        assert_values_close(diff, expected_diff, rtol=1e-2, atol=1e-2, name="awq bias add")

    def test_preserves_leading_dims(self, seed):
        """Input (..., K) should yield (..., N) regardless of rank."""
        B, T, N, K, group_size = 2, 3, 32, 64, 64
        dtype = torch.bfloat16
        num_groups = K // group_size

        qweight_vals = torch.randint(0, 16, (N, K), dtype=torch.int8)
        qweight = _pack_uint4_row_major(qweight_vals)
        wscales = torch.rand(num_groups, N, dtype=dtype) * 0.1 + 0.01
        wzeros = torch.zeros(num_groups, N, dtype=dtype)
        x = torch.randn(B, T, K, dtype=dtype) * 0.3

        with ck.use_backend("eager"):
            out = ck.gemv_awq_w4a16(x, qweight, wscales, wzeros, group_size=group_size)

        assert out.shape == (B, T, N)

    def test_k_not_divisible_by_group_raises(self):
        M, N, K, group_size = 4, 16, 96, 64  # 96 % 64 != 0
        qweight = torch.zeros(N, K // 2, dtype=torch.int8)
        wscales = torch.ones(2, N, dtype=torch.bfloat16)
        wzeros = torch.zeros(2, N, dtype=torch.bfloat16)
        x = torch.randn(M, K, dtype=torch.bfloat16)

        with ck.use_backend("eager"):
            with pytest.raises(ValueError, match="not divisible"):
                ck.gemv_awq_w4a16(x, qweight, wscales, wzeros, group_size=group_size)
