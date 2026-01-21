# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block-wise INT8 quantization layout for tensor cores.

This provides a QuantizedTensor layout for block-wise INT8 quantization,
following the same patterns as TensorCoreFP8Layout and TensorCoreMXFP8Layout.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .base import BaseLayoutParams, QuantizedLayout, dequantize_args, register_layout_op

if TYPE_CHECKING:
    from .base import QuantizedTensor

logger = logging.getLogger(__name__)


class BlockWiseINT8Layout(QuantizedLayout):
    """Block-wise INT8 quantization with per-block scaling.

    Uses asymmetric blocking:
    - Activations: 1D blocks along last dimension (K), block_size=128
    - Weights: 2D blocks along (M, N) dimensions, block_size=128

    Example:
        >>> x = torch.randn(512, 4096, device="cuda", dtype=torch.bfloat16)
        >>> qt = QuantizedTensor.from_float(x, "BlockWiseINT8Layout", is_weight=True)
        >>> qt.shape
        torch.Size([512, 4096])
        >>> dq = qt.dequantize()
        >>> torch.allclose(dq, x, rtol=0.1)
        True

    Note:
        Requires SM >= 7.5 (Turing) for INT8 tensor core support.
    """

    MIN_SM_VERSION = (7, 5)

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        """INT8 layout parameters.

        Inherits scale, orig_dtype, orig_shape from BaseLayoutParams.
        """
        block_size: int = 128
        is_weight: bool = False

        def _tensor_fields(self) -> list[str]:
            return ["scale"]

        def _validate_tensor_fields(self):
            pass

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        block_size: int = 128,
        is_weight: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Params]:
        """Quantize a tensor to INT8 with block-wise scaling.

        Args:
            tensor: Input tensor to quantize.
            block_size: Size of quantization blocks (default 128).
            is_weight: If True, use 2D blocking for weights. If False, use 1D for activations.
            **kwargs: Additional arguments (ignored).

        Returns:
            Tuple of (quantized_data, params).
        """
        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        # Import quantization function - Try Triton first, fall back to eager
        try:
            from comfy_kitchen.backends.triton.quantization import quantize_int8
            qdata, scale = quantize_int8(tensor, block_size=block_size, is_weight=is_weight)
        except (ImportError, RuntimeError):
            from comfy_kitchen.backends.eager.quantization import quantize_int8
            qdata, scale = quantize_int8(tensor, block_size=block_size, is_weight=is_weight)

        params = cls.Params(
            scale=scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
            block_size=block_size,
            is_weight=is_weight,
        )
        return qdata, params

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: Params) -> torch.Tensor:
        """Dequantize INT8 data back to original dtype.

        Args:
            qdata: Quantized INT8 data.
            params: Layout parameters including scale.

        Returns:
            Dequantized tensor.
        """
        # Try Triton first, fall back to eager
        try:
            from comfy_kitchen.backends.triton.quantization import dequantize_int8
            return dequantize_int8(
                qdata, params.scale,
                block_size=params.block_size,
                output_dtype=params.orig_dtype,
            )
        except (ImportError, RuntimeError):
            from comfy_kitchen.backends.eager.quantization import dequantize_int8
            return dequantize_int8(
                qdata, params.scale,
                block_size=params.block_size,
                output_dtype=params.orig_dtype,
            )

    @classmethod
    def get_plain_tensors(cls, qtensor: QuantizedTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract raw tensors for computation.

        Args:
            qtensor: Quantized tensor.

        Returns:
            Tuple of (quantized_data, scale).
        """
        return qtensor._qdata, qtensor._params.scale

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
        """Return key suffix → tensor mapping for serialization.

        Args:
            qdata: Quantized data.
            params: Layout parameters.

        Returns:
            Dictionary mapping suffix to tensor.
        """
        return {
            "": qdata,
            "_scale": params.scale,
        }


class TensorWiseINT8Layout(QuantizedLayout):
    """Tensor-wise INT8 quantization (from dxqb/OneTrainer).

    Simpler approach than block-wise:
    - Weights: Single scale per tensor
    - Activations: Per-row scales (dynamic quantization)

    Uses torch._int_mm/cuBLASLt IMMA for fast matmul.

    Example:
        >>> w = torch.randn(512, 4096, device="cuda", dtype=torch.bfloat16)
        >>> qt = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")
        >>> qt.shape
        torch.Size([512, 4096])

    Note:
        Requires SM >= 7.5 (Turing) for INT8 tensor core support.
    """

    MIN_SM_VERSION = (7, 5)

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        """Tensor-wise INT8 layout parameters.

        Inherits scale, orig_dtype, orig_shape from BaseLayoutParams.
        """
        is_weight: bool = True

        def _tensor_fields(self) -> list[str]:
            return ["scale"]

        def _validate_tensor_fields(self):
            pass

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        is_weight: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, Params]:
        """Quantize a tensor to INT8 with tensorwise or rowwise scaling.

        Args:
            tensor: Input tensor to quantize.
            is_weight: If True, use tensorwise scale. If False, use per-row.
            **kwargs: Additional arguments (ignored).

        Returns:
            Tuple of (quantized_data, params).
        """
        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        # Import from eager backend (works on both CPU and CUDA)
        from comfy_kitchen.backends.eager.quantization import (
            quantize_int8_tensorwise,
            quantize_int8_rowwise,
        )

        if is_weight:
            qdata, scale = quantize_int8_tensorwise(tensor)
        else:
            qdata, scale = quantize_int8_rowwise(tensor)

        params = cls.Params(
            scale=scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
            is_weight=is_weight,
        )
        return qdata, params

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: Params) -> torch.Tensor:
        """Dequantize INT8 data back to original dtype.

        Args:
            qdata: Quantized INT8 data.
            params: Layout parameters including scale.

        Returns:
            Dequantized tensor.
        """
        from comfy_kitchen.backends.eager.quantization import dequantize_int8_simple
        result = dequantize_int8_simple(qdata, params.scale)
        return result.to(params.orig_dtype)

    @classmethod
    def get_plain_tensors(cls, qtensor: QuantizedTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract raw tensors for computation.

        Args:
            qtensor: Quantized tensor.

        Returns:
            Tuple of (quantized_data, scale).
        """
        return qtensor._qdata, qtensor._params.scale

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
        """Return key suffix → tensor mapping for serialization.

        Args:
            qdata: Quantized data.
            params: Layout parameters.

        Returns:
            Dictionary mapping suffix to tensor.
        """
        return {
            "": qdata,
            "_scale": params.scale,
        }

    @classmethod
    def supports_fast_matmul(cls) -> bool:
        """Check if fast INT8 matmul is available."""
        if not torch.cuda.is_available():
            return False
        sm_major, sm_minor = torch.cuda.get_device_capability()
        return (sm_major, sm_minor) >= cls.MIN_SM_VERSION


# =============================================================================
# INT8 Matmul Operations
# =============================================================================

def _int8_scaled_mm(
    input_qdata: torch.Tensor,
    weight_qdata: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """INT8 scaled matmul using best available backend.

    Args:
        input_qdata: INT8 input data.
        weight_qdata: INT8 weight data (already in N, K format).
        scale_a: Input scales.
        scale_b: Weight scales.
        bias: Optional bias.
        out_dtype: Output dtype.

    Returns:
        Result of scaled matmul.
    """
    # Try Triton kernel first
    try:
        from comfy_kitchen.backends.triton.quantization import scaled_mm_int8
        return scaled_mm_int8(input_qdata, weight_qdata, scale_a, scale_b, bias, out_dtype or torch.bfloat16)
    except (ImportError, RuntimeError):
        pass

    # Fallback to eager backend
    try:
        from comfy_kitchen.backends.eager.quantization import scaled_mm_int8
        return scaled_mm_int8(input_qdata, weight_qdata, scale_a, scale_b, bias, out_dtype or torch.bfloat16)
    except (ImportError, RuntimeError):
        pass

    # Final fallback: dequantize and use standard matmul
    from comfy_kitchen.backends.eager.quantization import dequantize_int8
    a_fp = dequantize_int8(input_qdata, scale_a, 128, out_dtype or torch.bfloat16)
    b_fp = dequantize_int8(weight_qdata, scale_b, 128, out_dtype or torch.bfloat16)
    result = torch.nn.functional.linear(a_fp, b_fp, bias)
    return result.to(out_dtype) if out_dtype else result


@register_layout_op(torch.ops.aten.linear.default, BlockWiseINT8Layout)
def _handle_int8_linear(qt, args, kwargs):
    """INT8 linear: output = input @ weight.T + bias.

    Uses Triton INT8 GEMM kernel when both input and weight are
    INT8 QuantizedTensors.
    """
    from .base import QuantizedTensor

    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None

    # Fast path: both operands are INT8 QuantizedTensors
    if not (isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor)):
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))

    input_qdata, scale_a = BlockWiseINT8Layout.get_plain_tensors(input_tensor)
    weight_qdata, scale_b = BlockWiseINT8Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", input_tensor._params.orig_dtype)

    try:
        return _int8_scaled_mm(input_qdata, weight_qdata, scale_a, scale_b, bias, out_dtype)
    except (RuntimeError, TypeError) as e:
        logger.warning(f"INT8 scaled_mm failed: {e}, falling back to dequantization")
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))


@register_layout_op(torch.ops.aten.mm.default, BlockWiseINT8Layout)
def _handle_int8_mm(qt, args, kwargs):
    """INT8 matrix multiplication: output = a @ b."""
    from .base import QuantizedTensor

    a, b = args[0], args[1]

    if not (isinstance(a, QuantizedTensor) and isinstance(b, QuantizedTensor)):
        return torch.mm(*dequantize_args(args))

    a_qdata, scale_a = BlockWiseINT8Layout.get_plain_tensors(a)
    b_qdata, scale_b = BlockWiseINT8Layout.get_plain_tensors(b)
    out_dtype = kwargs.get("out_dtype", a._params.orig_dtype)

    try:
        # Note: mm expects b to NOT be transposed, but our kernel expects (N, K)
        # For mm, b is (K, N), so we need to transpose it
        return _int8_scaled_mm(a_qdata, b_qdata.t().contiguous(), scale_a, scale_b.t().contiguous(), None, out_dtype)
    except (RuntimeError, TypeError):
        return torch.mm(*dequantize_args(args))


@register_layout_op(torch.ops.aten.addmm.default, BlockWiseINT8Layout)
def _handle_int8_addmm(qt, args, kwargs):
    """INT8 addmm: output = bias + input @ weight."""
    from .base import QuantizedTensor

    bias, input_tensor, weight = args[0], args[1], args[2]

    if not (isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor)):
        return torch.addmm(*dequantize_args(args))

    input_qdata, scale_a = BlockWiseINT8Layout.get_plain_tensors(input_tensor)
    weight_qdata, scale_b = BlockWiseINT8Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", input_tensor._params.orig_dtype)

    try:
        # addmm decomposition: bias + input @ weight
        # weight is (K, N) for addmm, transpose to (N, K) for our kernel
        return _int8_scaled_mm(input_qdata, weight_qdata.t().contiguous(), scale_a, scale_b.t().contiguous(), bias, out_dtype)
    except (RuntimeError, TypeError):
        return torch.addmm(*dequantize_args(args))


# =============================================================================
# INT8 Shape Operations
# =============================================================================

@register_layout_op(torch.ops.aten.t.default, BlockWiseINT8Layout)
def _handle_int8_transpose(qt, args, kwargs):
    """Handle transpose for INT8 tensors.

    For weights with 2D blocking, we need to transpose the scale tensor too.
    """
    from .base import QuantizedTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.t.default(*args, **kwargs)

    # Transpose the quantized data
    new_qdata = input_tensor._qdata.t().contiguous()

    # Transpose scale if it's 2D (weight tensor)
    old_scale = input_tensor._params.scale
    if old_scale.dim() == 2:
        new_scale = old_scale.t().contiguous()
    else:
        new_scale = old_scale

    old_shape = input_tensor._params.orig_shape
    new_params = BlockWiseINT8Layout.Params(
        scale=new_scale,
        orig_dtype=input_tensor._params.orig_dtype,
        orig_shape=(old_shape[1], old_shape[0]) if len(old_shape) == 2 else old_shape,
        block_size=input_tensor._params.block_size,
        is_weight=input_tensor._params.is_weight,
    )
    return QuantizedTensor(new_qdata, "BlockWiseINT8Layout", new_params)


def _make_int8_view_handler(aten_op):
    """Factory for shape-changing operations that preserve INT8 data.

    INT8 is not packed (1:1 element mapping), so view/reshape work directly.
    However, for block-wise quantization, view operations may break block alignment.
    """
    from .base import QuantizedTensor

    def handler(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return aten_op(*args, **kwargs)

        # For activation tensors (1D blocking along K), view is generally safe
        # For weight tensors (2D blocking), view may break alignment
        if input_tensor._params.is_weight:
            logger.warning(
                f"INT8 view operation on weight tensor may break block alignment. "
                f"Falling back to dequantization."
            )
            return aten_op(input_tensor.dequantize(), *args[1:], **kwargs)

        # Apply view to quantized data
        new_qdata = aten_op(input_tensor._qdata, *args[1:], **kwargs)

        # Scale shape depends on the new data shape
        # For now, keep existing scale (assumes view preserves last dimension blocks)
        new_params = BlockWiseINT8Layout.Params(
            scale=input_tensor._params.scale,
            orig_dtype=input_tensor._params.orig_dtype,
            orig_shape=tuple(new_qdata.shape),
            block_size=input_tensor._params.block_size,
            is_weight=False,  # After view, treat as activation
        )
        return QuantizedTensor(new_qdata, "BlockWiseINT8Layout", new_params)

    return handler


# Register view operation
register_layout_op(torch.ops.aten.view.default, BlockWiseINT8Layout)(
    _make_int8_view_handler(torch.ops.aten.view.default)
)
