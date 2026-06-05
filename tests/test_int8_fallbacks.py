# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for INT8 fallbacks (small M and per-channel) in Eager backend."""

import pytest
import torch
import comfy_kitchen as ck
from comfy_kitchen.backends.eager.quantization import quantize_int8_tensorwise, quantize_int8_rowwise

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fallback tests")
class TestINT8Fallbacks:
    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)

    def test_small_m_padding(self):
        """Test that int8_linear works with M < 16 (e.g. M=1) using padding."""
        m, k, n = 1, 128, 64
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        w_int8, w_scale = quantize_int8_tensorwise(w)

        with ck.registry.use_backend("eager"):
            # This would previously crash with RuntimeError: self.size(0) needs to be greater than 16, but got 1
            out = ck.int8_linear(x, w_int8, w_scale, out_dtype=torch.bfloat16)

        assert out.shape == (m, n)
        assert out.dtype == torch.bfloat16

        # Verify correctness against float linear
        ref_out = torch.nn.functional.linear(x, w)
        # Higher tolerance for INT8 vs BF16
        torch.testing.assert_close(out, ref_out, rtol=0.2, atol=0.2)

    def test_per_channel_requantization(self):
        """Test that int8_linear works with per-channel weights via re-quantization."""
        m, k, n = 32, 128, 64
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

        # Create per-channel (rowwise) quantized weight
        w_int8, w_scale = quantize_int8_rowwise(w)
        assert w_scale.numel() == n

        with ck.registry.use_backend("eager"):
            # This triggers the per-channel -> tensorwise conversion
            out = ck.int8_linear(x, w_int8, w_scale, out_dtype=torch.bfloat16)

        assert out.shape == (m, n)
        assert out.dtype == torch.bfloat16

        # Verify correctness
        ref_out = torch.nn.functional.linear(x, w)
        # Re-quantization from per-channel to tensor-wise is lossy, so we use higher tolerance
        torch.testing.assert_close(out, ref_out, rtol=0.5, atol=0.5)

    def test_small_m_and_per_channel(self):
        """Test combined fallback: M=1 AND per-channel weights."""
        m, k, n = 1, 128, 64
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

        w_int8, w_scale = quantize_int8_rowwise(w)

        with ck.registry.use_backend("eager"):
            out = ck.int8_linear(x, w_int8, w_scale, out_dtype=torch.bfloat16)

        assert out.shape == (m, n)
        ref_out = torch.nn.functional.linear(x, w)
        torch.testing.assert_close(out, ref_out, rtol=0.5, atol=0.5)

    def test_mm_int8_small_m(self):
        """Test mm_int8 padding for small M."""
        from comfy_kitchen.backends.eager.quantization import mm_int8

        m, k, n = 1, 128, 64
        a = torch.randint(-128, 127, (m, k), device="cuda", dtype=torch.int8)
        b = torch.randint(-128, 127, (k, n), device="cuda", dtype=torch.int8)

        # This would previously crash
        out = mm_int8(a, b)

        assert out.shape == (m, n)
        assert out.dtype == torch.int32

        ref_out = (a.float() @ b.float()).to(torch.int32)
        torch.testing.assert_close(out, ref_out)
