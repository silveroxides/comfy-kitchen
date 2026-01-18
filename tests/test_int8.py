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
# INT8 Activation Quantization Tests
# =============================================================================


class TestQuantizeINT8Blockwise:
    """INT8 block-wise activation quantization tests."""

    @pytest.fixture
    def capable_backends(self, device):
        """Get backends capable of running on current device."""
        backends = get_capable_backends("quantize_int8_blockwise", device)
        if not backends:
            pytest.skip(f"No backend supports quantize_int8_blockwise on {device}")
        return backends

    @pytest.mark.parametrize("shape", [
        (128, 256),
        (512, 1024),
        (1, 128),
        (64, 128, 256),  # 3D tensor
    ])
    def test_quantize_int8_blockwise_shapes(self, capable_backends, device, seed, shape):
        """Test INT8 activation quantization with various shapes."""
        for backend_name in capable_backends:
            if backend_name == "triton" and device == "cpu":
                continue

            if backend_name == "triton":
                from comfy_kitchen.backends.triton.quantization import (
                    quantize_int8_blockwise,
                )
            else:
                from comfy_kitchen.backends.eager.quantization import (
                    quantize_int8_blockwise,
                )

            x = torch.randn(shape, device=device, dtype=torch.float32)
            qx, scale = quantize_int8_blockwise(x, block_size=128)

            assert qx.shape == x.shape, f"{backend_name}: shape mismatch"
            assert qx.dtype == torch.int8, f"{backend_name}: expected int8"
            expected_scale_shape = list(shape)
            expected_scale_shape[-1] = shape[-1] // 128
            assert list(scale.shape) == expected_scale_shape, f"{backend_name}: scale shape mismatch"

    def test_quantize_dequantize_roundtrip(self, capable_backends, device, seed):
        """Test that quantize/dequantize roundtrip preserves values reasonably."""
        for backend_name in capable_backends:
            if backend_name == "triton" and device == "cpu":
                continue

            if backend_name == "triton":
                from comfy_kitchen.backends.triton.quantization import (
                    dequantize_int8_blockwise,
                    quantize_int8_blockwise,
                )
            else:
                from comfy_kitchen.backends.eager.quantization import (
                    dequantize_int8_blockwise,
                    quantize_int8_blockwise,
                )

            x = torch.randn(256, 512, device=device, dtype=torch.float32)
            qx, scale = quantize_int8_blockwise(x, block_size=128)
            dx = dequantize_int8_blockwise(qx, scale, block_size=128, output_dtype=torch.float32)

            # INT8 quantization should have small error (<5% relative)
            rel_error = (x - dx).abs() / (x.abs() + 1e-8)
            assert rel_error.mean() < 0.05, f"{backend_name}: mean relative error too high: {rel_error.mean()}"


class TestQuantizeINT8Weight:
    """INT8 2D weight quantization tests."""

    @pytest.fixture
    def capable_backends(self, device):
        backends = get_capable_backends("quantize_int8_weight", device)
        if not backends:
            pytest.skip(f"No backend supports quantize_int8_weight on {device}")
        return backends

    @pytest.mark.parametrize("m,n", [
        (256, 512),
        (512, 1024),
        (128, 128),
    ])
    def test_quantize_int8_weight_shapes(self, capable_backends, device, seed, m, n):
        """Test INT8 weight quantization with various shapes."""
        for backend_name in capable_backends:
            if backend_name == "triton" and device == "cpu":
                continue

            if backend_name == "triton":
                from comfy_kitchen.backends.triton.quantization import (
                    quantize_int8_weight,
                )
            else:
                from comfy_kitchen.backends.eager.quantization import (
                    quantize_int8_weight,
                )

            x = torch.randn(m, n, device=device, dtype=torch.float32)
            qx, scale = quantize_int8_weight(x, block_size=128)

            assert qx.shape == (m, n), f"{backend_name}: shape mismatch"
            assert qx.dtype == torch.int8, f"{backend_name}: expected int8"
            assert scale.shape == (m // 128, n // 128), f"{backend_name}: scale shape mismatch"

    def test_weight_quantize_dequantize_roundtrip(self, capable_backends, device, seed):
        """Test weight quantize/dequantize roundtrip."""
        for backend_name in capable_backends:
            if backend_name == "triton" and device == "cpu":
                continue

            if backend_name == "triton":
                from comfy_kitchen.backends.triton.quantization import (
                    dequantize_int8_weight,
                    quantize_int8_weight,
                )
            else:
                from comfy_kitchen.backends.eager.quantization import (
                    dequantize_int8_weight,
                    quantize_int8_weight,
                )

            x = torch.randn(256, 512, device=device, dtype=torch.float32)
            qx, scale = quantize_int8_weight(x, block_size=128)
            dx = dequantize_int8_weight(qx, scale, block_size=128, output_dtype=torch.float32)

            rel_error = (x - dx).abs() / (x.abs() + 1e-8)
            assert rel_error.mean() < 0.05, f"{backend_name}: mean relative error too high"


# =============================================================================
# INT8 GEMM Tests
# =============================================================================


class TestINT8GEMM:
    """INT8 matrix multiplication tests."""

    @pytest.fixture
    def capable_backends(self, device):
        backends = get_capable_backends("int8_gemm", device)
        if not backends:
            pytest.skip(f"No backend supports int8_gemm on {device}")
        return backends

    @pytest.mark.parametrize("m,k,n", [
        (256, 512, 256),
        (128, 256, 128),
    ])
    def test_int8_gemm_correctness(self, capable_backends, device, seed, m, k, n):
        """Test INT8 GEMM produces correct results."""
        for backend_name in capable_backends:
            if backend_name == "triton" and device == "cpu":
                continue

            if backend_name == "triton":
                from comfy_kitchen.backends.triton.quantization import (
                    int8_gemm,
                    quantize_int8_blockwise,
                    quantize_int8_weight,
                )
            else:
                from comfy_kitchen.backends.eager.quantization import (
                    int8_gemm,
                    quantize_int8_blockwise,
                    quantize_int8_weight,
                )

            # Create test data
            a = torch.randn(m, k, device=device, dtype=torch.float32)
            b = torch.randn(n, k, device=device, dtype=torch.float32)

            # Reference result
            ref = torch.nn.functional.linear(a, b)

            # Quantized computation
            qa, a_s = quantize_int8_blockwise(a, block_size=128)
            qb, b_s = quantize_int8_weight(b, block_size=128)
            result = int8_gemm(qa, a_s, qb, b_s, out_dtype=torch.float32)

            assert result.shape == ref.shape, f"{backend_name}: shape mismatch"

            # INT8 GEMM should have reasonable accuracy
            rel_error = (result - ref).abs() / (ref.abs() + 1e-6)
            assert rel_error.mean() < 0.15, f"{backend_name}: mean relative error too high: {rel_error.mean()}"


class TestINT8AddMM:
    """INT8 matrix multiplication with bias tests."""

    @pytest.fixture
    def capable_backends(self, device):
        backends = get_capable_backends("int8_addmm", device)
        if not backends:
            pytest.skip(f"No backend supports int8_addmm on {device}")
        return backends

    def test_int8_addmm_with_bias(self, capable_backends, device, seed):
        """Test INT8 addmm with bias produces correct results."""
        m, k, n = 256, 512, 256

        for backend_name in capable_backends:
            if backend_name == "triton" and device == "cpu":
                continue

            if backend_name == "triton":
                from comfy_kitchen.backends.triton.quantization import (
                    int8_addmm,
                    quantize_int8_blockwise,
                    quantize_int8_weight,
                )
            else:
                from comfy_kitchen.backends.eager.quantization import (
                    int8_addmm,
                    quantize_int8_blockwise,
                    quantize_int8_weight,
                )

            # Create test data
            a = torch.randn(m, k, device=device, dtype=torch.float32)
            b = torch.randn(n, k, device=device, dtype=torch.float32)
            bias = torch.randn(n, device=device, dtype=torch.float32)

            # Reference result
            ref = torch.nn.functional.linear(a, b, bias)

            # Quantized computation
            qa, a_s = quantize_int8_blockwise(a, block_size=128)
            qb, b_s = quantize_int8_weight(b, block_size=128)
            result = int8_addmm(qa, a_s, qb, b_s, bias=bias, out_dtype=torch.float32)

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

    def test_layout_activation_quantization(self, cuda_available):
        """Test BlockWiseINT8Layout for activations (not weights)."""
        from comfy_kitchen.tensor import QuantizedTensor

        x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(x, "BlockWiseINT8Layout", is_weight=False)

        assert qt._params.is_weight is False
        assert qt._params.block_size == 128

        dq = qt.dequantize()
        rel_error = (x - dq).abs() / (x.abs() + 1e-8)
        assert rel_error.mean() < 0.05


# =============================================================================
# Cross-Backend Consistency Tests
# =============================================================================


class TestINT8CrossBackend:
    """Test consistency between Triton and eager backends."""

    @pytest.fixture
    def both_backends_available(self, device):
        triton_available = "triton" in get_capable_backends("quantize_int8_blockwise", device)
        eager_available = "eager" in get_capable_backends("quantize_int8_blockwise", device)
        if not (triton_available and eager_available and device == "cuda"):
            pytest.skip("Need both triton and eager backends on CUDA")

    def test_activation_quantize_consistency(self, both_backends_available, device, seed):
        """Test that triton and eager produce consistent activation quantization."""
        from comfy_kitchen.backends.eager.quantization import (
            quantize_int8_blockwise as eager_quant,
        )
        from comfy_kitchen.backends.triton.quantization import (
            quantize_int8_blockwise as triton_quant,
        )

        x = torch.randn(256, 512, device=device, dtype=torch.float32)

        eager_qx, eager_s = eager_quant(x, block_size=128)
        triton_qx, triton_s = triton_quant(x, block_size=128)

        # Quantized values should be identical or very close
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

    def test_weight_quantize_consistency(self, both_backends_available, device, seed):
        """Test that triton and eager produce consistent weight quantization."""
        from comfy_kitchen.backends.eager.quantization import (
            quantize_int8_weight as eager_quant,
        )
        from comfy_kitchen.backends.triton.quantization import (
            quantize_int8_weight as triton_quant,
        )

        x = torch.randn(256, 512, device=device, dtype=torch.float32)

        eager_qx, eager_s = eager_quant(x, block_size=128)
        triton_qx, triton_s = triton_quant(x, block_size=128)

        assert_values_close(
            eager_qx.float(), triton_qx.float(),
            rtol=0.0, atol=1.0,
            name="quantized weight (triton vs eager)"
        )

        assert_values_close(
            eager_s, triton_s,
            rtol=1e-4, atol=1e-6,
            name="weight scales (triton vs eager)"
        )
