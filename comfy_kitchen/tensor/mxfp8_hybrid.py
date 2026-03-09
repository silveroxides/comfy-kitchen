# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 Hybrid layout with tensorwise fallback for Ada GPUs.

This layout extends TensorCoreMXFP8Layout to additionally store a tensorwise
scale that can be used for FP8 matmul on SM 8.9 (Ada) GPUs via torch._scaled_mm,
instead of falling back to full dequantization.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .base import BaseLayoutParams, dequantize_args, get_cuda_capability, register_layout_op
from .mxfp8 import TensorCoreMXFP8Layout

if TYPE_CHECKING:
    from .base import QuantizedTensor

logger = logging.getLogger(__name__)


def _compute_tensorwise_scale(block_scales: torch.Tensor) -> torch.Tensor:
    """Compute optimal tensorwise scale from E8M0 block scales.

    Uses the maximum block scale to ensure all values fit within FP8 range.
    This is conservative but guarantees no overflow.

    Args:
        block_scales: E8M0 block scales in swizzled layout (float8_e8m0fnu)

    Returns:
        Single float32 tensorwise scale (scalar tensor)

    Raises:
        ValueError: If block_scales is not 2D or empty
    """
    from comfy_kitchen.float_utils import e8m0_to_f32

    # Input validation
    if block_scales.dim() != 2:
        raise ValueError(f"block_scales must be 2D, got {block_scales.dim()}D")
    if block_scales.numel() == 0:
        raise ValueError("block_scales cannot be empty")

    # Convert E8M0 to float32 (E8M0 is stored as uint8 exponents)
    scales_uint8 = block_scales.view(torch.uint8)
    scales_f32 = e8m0_to_f32(scales_uint8)

    # Get max scale (handles padding gracefully since max is still valid)
    max_scale = scales_f32.max()

    # Handle edge case: all-zero scales (shouldn't happen but be safe)
    if max_scale == 0:
        logger.warning("All block scales are zero, using scale=1.0 as fallback")
        max_scale = torch.tensor(1.0, device=block_scales.device, dtype=torch.float32)

    result = max_scale.to(torch.float32).reshape(())
    logger.debug(f"Computed tensorwise scale: {result.item():.6f} from {scales_f32.numel()} block scales")
    return result


class HybridMXFP8Layout(TensorCoreMXFP8Layout):
    """MXFP8 layout with optional tensorwise scale for Ada (SM 8.9) fallback.

    Extends TensorCoreMXFP8Layout to compute and store an additional tensorwise
    scale during quantization. At runtime:
    - SM >= 10.0 (Blackwell): Uses native MXFP8 block-wise scaled matmul
    - SM 8.9 (Ada) + scalar present: Uses torch._scaled_mm with tensorwise scale
    - Otherwise: Falls back to dequantization

    Storage format:
    - "": FP8 quantized data
    - "_scale": E8M0 block scales (inherited from TensorCoreMXFP8Layout)
    - "_scalar": Optional tensorwise scale for Ada fallback

    Note:
        Models without "_scalar" are fully backward compatible - they will
        use dequantization fallback on pre-Blackwell GPUs as before.
    """

    MIN_SM_VERSION = (8, 9)  # Ada compatible (native MXFP8 needs SM 10.0)

    @dataclass(frozen=True)
    class Params(TensorCoreMXFP8Layout.Params):
        """Hybrid MXFP8 parameters with optional tensorwise scale."""
        scalar: torch.Tensor | None = None

        def _tensor_fields(self) -> list[str]:
            fields = ["scale"]
            if self.scalar is not None:
                fields.append("scalar")
            return fields

        def _validate_tensor_fields(self):
            """Validate scalar dtype and shape."""
            super()._validate_tensor_fields()
            if self.scalar is not None:
                # Ensure scalar is float32 scalar
                ts = self.scalar
                if ts.numel() != 1:
                    raise ValueError(f"scalar must be scalar, got shape {ts.shape}")
                if ts.dtype != torch.float32:
                    object.__setattr__(self, "scalar", ts.to(torch.float32))

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, Params]:
        """Quantize tensor to MXFP8 with additional tensorwise scale.

        Args:
            tensor: Input tensor (2D, shape M x K)
            **kwargs: Passed to parent quantize()

        Returns:
            Tuple of (quantized_fp8_tensor, params_with_scalar)
        """
        # Use parent quantization to get FP8 data and block scales
        qdata, parent_params = TensorCoreMXFP8Layout.quantize(tensor, **kwargs)

        # Compute tensorwise scale from block scales
        scalar = _compute_tensorwise_scale(parent_params.scale)

        # Create hybrid params with scalar
        params = cls.Params(
            scale=parent_params.scale,
            orig_dtype=parent_params.orig_dtype,
            orig_shape=parent_params.orig_shape,
            transposed=parent_params.transposed,
            scalar=scalar,
        )
        return qdata, params

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
        """Return key suffix → tensor mapping for serialization.

        Includes "_scalar" suffix for tensorwise scale when present.
        """
        result = {"": qdata, "_scale": params.scale}
        if params.scalar is not None:
            result["_scalar"] = params.scalar
        return result


# =============================================================================
# Hybrid MXFP8 Scaled Matmul
# =============================================================================

def _hybrid_mxfp8_scaled_mm(
    a_qdata: torch.Tensor,
    b_qdata: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    scalar_a: torch.Tensor | None = None,
    scalar_b: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Hybrid MXFP8 scaled matmul with Ada fallback.

    Dispatches to appropriate path based on SM version:
    - SM >= 10.0: Native MXFP8 block-wise matmul
    - SM 8.9 + tensor scales: Tensorwise FP8 matmul via torch._scaled_mm
    - Otherwise: Raises RuntimeError (caller should dequantize)

    Args:
        a_qdata: FP8 input data
        b_qdata: FP8 weight data
        scale_a: E8M0 block scales for input
        scale_b: E8M0 block scales for weight
        scalar_a: Optional tensorwise scale for input
        scalar_b: Optional tensorwise scale for weight
        bias: Optional bias
        out_dtype: Output dtype

    Returns:
        Result tensor
    """
    cap = get_cuda_capability()
    if cap is None:
        raise RuntimeError("CUDA not available")

    # Path 1: Native MXFP8 on Blackwell (SM >= 10.0)
    if cap >= (10, 0):
        import comfy_kitchen as ck
        return ck.scaled_mm_mxfp8(
            a_qdata, b_qdata,
            block_scale_a=scale_a,
            block_scale_b=scale_b,
            bias=bias,
            out_dtype=out_dtype,
        )

    # Path 2: Tensorwise FP8 on Ada (SM 8.9)
    if cap >= (8, 9) and scalar_a is not None and scalar_b is not None:
        logger.debug("Using tensorwise FP8 fallback for Ada GPU")
        # Use torch._scaled_mm with tensorwise scales
        # Note: b needs to be transposed for linear semantics (a @ b.T)
        output = torch._scaled_mm(
            a_qdata.contiguous(),
            b_qdata.t().contiguous(),
            scale_a=scalar_a,
            scale_b=scalar_b,
            out_dtype=out_dtype,
        )
        # Handle tuple return from older PyTorch versions
        if isinstance(output, tuple):
            output = output[0]
        if bias is not None:
            output = output + bias
        return output

    # Path 3: No fast path available
    raise RuntimeError(
        f"No fast matmul path available for SM {cap}. "
        f"scalar_a={scalar_a is not None}, scalar_b={scalar_b is not None}"
    )


# =============================================================================
# Dispatch Handlers
# =============================================================================

@register_layout_op(torch.ops.aten.t.default, HybridMXFP8Layout)
def _handle_hybrid_mxfp8_transpose(qt, args, kwargs):
    """Handle transpose as a logical flag flip for HybridMXFP8."""
    from .base import QuantizedTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.t.default(*args, **kwargs)

    old_shape = input_tensor._params.orig_shape
    new_params = HybridMXFP8Layout.Params(
        scale=input_tensor._params.scale,
        orig_dtype=input_tensor._params.orig_dtype,
        orig_shape=(old_shape[1], old_shape[0]),
        transposed=not input_tensor._params.transposed,
        scalar=input_tensor._params.scalar,
    )
    return QuantizedTensor(input_tensor._qdata, "HybridMXFP8Layout", new_params)


@register_layout_op(torch.ops.aten.mm.default, HybridMXFP8Layout)
def _handle_hybrid_mxfp8_mm(qt, args, kwargs):
    """Hybrid MXFP8 mm: requires b to be logically transposed (from .t() call)."""
    from .base import QuantizedTensor

    a, b = args[0], args[1]

    if not (isinstance(a, QuantizedTensor) and isinstance(b, QuantizedTensor)):
        return torch.mm(*dequantize_args(args))
    if a._qdata.dim() != 2:
        return torch.mm(*dequantize_args(args))

    a_transposed = getattr(a._params, "transposed", False)
    b_transposed = getattr(b._params, "transposed", False)

    if a_transposed or not b_transposed:
        return torch.mm(*dequantize_args(args))

    a_qdata, scale_a = HybridMXFP8Layout.get_plain_tensors(a)
    b_qdata, scale_b = HybridMXFP8Layout.get_plain_tensors(b)
    scalar_a = a._params.scalar
    scalar_b = b._params.scalar
    out_dtype = kwargs.get("out_dtype", a._params.orig_dtype)

    try:
        result = _hybrid_mxfp8_scaled_mm(
            a_qdata, b_qdata, scale_a, scale_b,
            scalar_a, scalar_b,
            out_dtype=out_dtype,
        )
        # Slice to original shape if padded
        orig_m = a._params.orig_shape[0]
        orig_n = b._params.orig_shape[1]
        if result.shape[0] != orig_m or result.shape[1] != orig_n:
            result = result[:orig_m, :orig_n]
        return result
    except (RuntimeError, TypeError) as e:
        logger.warning(f"HybridMXFP8 mm failed: {e}")
        return torch.mm(*dequantize_args(args))


@register_layout_op(torch.ops.aten.addmm.default, HybridMXFP8Layout)
def _handle_hybrid_mxfp8_addmm(qt, args, kwargs):
    """Hybrid MXFP8 addmm: bias + input @ weight.T."""
    from .base import QuantizedTensor

    bias, mat1, mat2 = args[0], args[1], args[2]

    if not (isinstance(mat1, QuantizedTensor) and isinstance(mat2, QuantizedTensor)):
        return torch.addmm(*dequantize_args((bias, mat1, mat2)))
    if mat1._qdata.dim() != 2:
        return torch.addmm(*dequantize_args((bias, mat1, mat2)))

    input_transposed = getattr(mat1._params, "transposed", False)
    weight_transposed = getattr(mat2._params, "transposed", False)

    if input_transposed or not weight_transposed:
        return torch.addmm(*dequantize_args((bias, mat1, mat2)))

    input_qdata, scale_a = HybridMXFP8Layout.get_plain_tensors(mat1)
    weight_qdata, scale_b = HybridMXFP8Layout.get_plain_tensors(mat2)
    scalar_a = mat1._params.scalar
    scalar_b = mat2._params.scalar
    out_dtype = mat1._params.orig_dtype

    try:
        result = _hybrid_mxfp8_scaled_mm(
            input_qdata, weight_qdata, scale_a, scale_b,
            scalar_a, scalar_b,
            bias=bias, out_dtype=out_dtype,
        )
        orig_m = mat1._params.orig_shape[0]
        orig_n = mat2._params.orig_shape[1]
        if result.shape[0] != orig_m or result.shape[1] != orig_n:
            result = result[:orig_m, :orig_n]
        return result
    except (RuntimeError, TypeError) as e:
        logger.warning(f"HybridMXFP8 addmm failed: {e}")
        return torch.addmm(*dequantize_args((bias, mat1, mat2)))


@register_layout_op(torch.ops.aten.linear.default, HybridMXFP8Layout)
def _handle_hybrid_mxfp8_linear(qt, args, kwargs):
    """Hybrid MXFP8 linear: input @ weight.T + bias."""
    from .base import QuantizedTensor

    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None

    if not (isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor)):
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))
    if input_tensor._qdata.dim() != 2:
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))

    if getattr(input_tensor._params, "transposed", False) or getattr(weight._params, "transposed", False):
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))

    input_qdata, scale_a = HybridMXFP8Layout.get_plain_tensors(input_tensor)
    weight_qdata, scale_b = HybridMXFP8Layout.get_plain_tensors(weight)
    scalar_a = input_tensor._params.scalar
    scalar_b = weight._params.scalar
    out_dtype = kwargs.get("out_dtype", input_tensor._params.orig_dtype)

    try:
        result = _hybrid_mxfp8_scaled_mm(
            input_qdata, weight_qdata, scale_a, scale_b,
            scalar_a, scalar_b,
            bias=bias, out_dtype=out_dtype,
        )
        orig_m = input_tensor._params.orig_shape[0]
        orig_n = weight._params.orig_shape[0]
        if result.shape[0] != orig_m or result.shape[1] != orig_n:
            result = result[:orig_m, :orig_n]
        return result
    except (RuntimeError, TypeError) as e:
        logger.warning(f"HybridMXFP8 linear failed: {e}")
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))
