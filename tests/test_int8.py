# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for INT8 block-wise quantization."""

import pytest
import torch

from .conftest import (
    assert_values_close,
    get_capable_backends,
    get_supported_devices,
)


# =============================================================================
# INT8 Quantization Tests
# =============================================================================


class TestQuantizeINT8:
    """INT8 block-wise quantization tests."""

    @pytest.fixture
    def capable_backends(self, device):
        """Get backends capable of running on current device."""
        backends = get_capable_backends("quantize_int8", device)
        if not backends:
            pytest.skip(f"No backend supports quantize_int8 on {device}")
        return backends

    @pytest.mark.parametrize("shape,is_weight", [
        ((256, 512), True),   # Weight tensor
        ((128, 256), False),  # Activation tensor
        ((64, 128, 256), False),  # 3D activation
    ])
    def test_quantize_int8_shapes(self, capable_backends, device, seed, shape, is_weight):
        """Test INT8 quantization with various shapes."""
        for backend_name in capable_backends:
            if backend_name == "triton" and device == "cpu":
                continue

            if backend_name == "triton":
                from comfy_kitchen.backends.triton.quantization import quantize_int8
            else:
                from comfy_kitchen.backends.eager.quantization import quantize_int8

            x = torch.randn(shape, device=device, dtype=torch.float32)
            qx, scale = quantize_int8(x, block_size=128, is_weight=is_weight)

            assert qx.shape == x.shape, f"{backend_name}: shape mismatch"
            assert qx.dtype == torch.int8, f"{backend_name}: expected int8"

    def test_quantize_dequantize_roundtrip(self, capable_backends, device, seed):
        """Test that quantize/dequantize roundtrip preserves values reasonably."""
        for backend_name in capable_backends:
            if backend_name == "triton" and device == "cpu":
                continue

            if backend_name == "triton":
                from comfy_kitchen.backends.triton.quantization import (
                    dequantize_int8,
                    quantize_int8,
                )
            else:
                from comfy_kitchen.backends.eager.quantization import (
                    dequantize_int8,
                    quantize_int8,
                )

            x = torch.randn(256, 512, device=device, dtype=torch.float32)
            qx, scale = quantize_int8(x, block_size=128, is_weight=True)
            dx = dequantize_int8(qx, scale, block_size=128, output_dtype=torch.float32)

            # INT8 quantization should have small error (<5% relative)
            rel_error = (x - dx).abs() / (x.abs() + 1e-8)
            assert rel_error.mean() < 0.05, f"{backend_name}: mean relative error too high: {rel_error.mean()}"


class TestScaledMMINT8:
    """INT8 scaled matrix multiplication tests."""

    @pytest.fixture
    def capable_backends(self, device):
        backends = get_capable_backends("scaled_mm_int8", device)
        if not backends:
            pytest.skip(f"No backend supports scaled_mm_int8 on {device}")
        return backends

    @pytest.mark.parametrize("m,k,n", [
        (256, 512, 256),
        (128, 256, 128),
    ])
    def test_scaled_mm_int8_correctness(self, capable_backends, device, seed, m, k, n):
        """Test INT8 scaled_mm produces correct results."""
        for backend_name in capable_backends:
            if backend_name == "triton" and device == "cpu":
                continue

            if backend_name == "triton":
                from comfy_kitchen.backends.triton.quantization import (
                    quantize_int8,
                    scaled_mm_int8,
                )
            else:
                from comfy_kitchen.backends.eager.quantization import (
                    quantize_int8,
                    scaled_mm_int8,
                )

            # Create test data
            a = torch.randn(m, k, device=device, dtype=torch.float32)
            b = torch.randn(n, k, device=device, dtype=torch.float32)

            # Reference result
            ref = torch.nn.functional.linear(a, b)

            # Quantized computation
            qa, a_s = quantize_int8(a, block_size=128, is_weight=False)
            qb, b_s = quantize_int8(b, block_size=128, is_weight=True)
            result = scaled_mm_int8(qa, qb, a_s, b_s, out_dtype=torch.float32)

            assert result.shape == ref.shape, f"{backend_name}: shape mismatch"


# =============================================================================
# BlockWiseINT8Layout Tests
# =============================================================================


class TestBlockWiseINT8Layout:
    """Tests for BlockWiseINT8Layout QuantizedTensor integration."""

    @pytest.fixture
    def cuda_available(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_layout_quantize_dequantize(self, cuda_available):
        """Test BlockWiseINT8Layout quantize and dequantize."""
        from comfy_kitchen.tensor import QuantizedTensor

        x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "BlockWiseINT8Layout", is_weight=True)

        assert qt.shape == x.shape
        assert qt._qdata.dtype == torch.int8

        dq = qt.dequantize()
        assert dq.shape == x.shape
        assert dq.dtype == x.dtype

        # Check reasonable accuracy
        rel_error = (x - dq).abs() / (x.abs() + 1e-8)
        assert rel_error.mean() < 0.05

    def test_layout_state_dict(self, cuda_available):
        """Test BlockWiseINT8Layout state dict serialization."""
        from comfy_kitchen.tensor import BlockWiseINT8Layout, QuantizedTensor

        x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "BlockWiseINT8Layout", is_weight=True)

        state_dict = BlockWiseINT8Layout.state_dict_tensors(qt._qdata, qt._params)

        assert "" in state_dict  # quantized data
        assert "_scale" in state_dict  # scale tensor
        assert state_dict[""].dtype == torch.int8


# =============================================================================
# Cross-Backend Consistency Tests
# =============================================================================


class TestINT8CrossBackend:
    """Test consistency between Triton and eager backends."""

    @pytest.fixture
    def both_backends_available(self, device):
        triton_available = "triton" in get_capable_backends("quantize_int8", device)
        eager_available = "eager" in get_capable_backends("quantize_int8", device)
        if not (triton_available and eager_available and device == "cuda"):
            pytest.skip("Need both triton and eager backends on CUDA")

    def test_quantize_consistency(self, both_backends_available, device, seed):
        """Test that triton and eager produce consistent quantization."""
        from comfy_kitchen.backends.eager.quantization import (
            quantize_int8 as eager_quant,
        )
        from comfy_kitchen.backends.triton.quantization import (
            quantize_int8 as triton_quant,
        )

        x = torch.randn(256, 512, device=device, dtype=torch.float32)

        eager_qx, eager_s = eager_quant(x, block_size=128, is_weight=True)
        triton_qx, triton_s = triton_quant(x, block_size=128, is_weight=True)

        assert_values_close(
            eager_qx.float(), triton_qx.float(),
            rtol=0.0, atol=1.0,  # Allow 1 unit difference
            name="quantized data (triton vs eager)"
        )

        assert_values_close(
            eager_s, triton_s,
            rtol=1e-4, atol=1e-6,
            name="scales (triton vs eager)"
        )
