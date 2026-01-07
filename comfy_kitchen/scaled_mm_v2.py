import logging

import torch
from packaging import version
from typing import Optional


_TORCH_VERSION = version.parse(torch.__version__.split("+")[0])  # Remove git hash suffix
TORCH_2_10 = version.parse("2.10.0")
_HAS_SCALED_MM_V2 = hasattr(torch.nn.functional, "scaled_mm")

if _HAS_SCALED_MM_V2:
    from torch.nn.functional import ScalingType, SwizzleType
else:
    # Dummy types for older PyTorch versions
    class ScalingType:
        TensorWise = "TensorWise"
        BlockWise1x16 = "BlockWise1x16"

    class SwizzleType:
        NO_SWIZZLE = "NO_SWIZZLE"
        SWIZZLE_32_4_4 = "SWIZZLE_32_4_4"


def has_scaled_mm_v2() -> bool:
    return _HAS_SCALED_MM_V2

def scaled_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    scale_result: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if has_scaled_mm_v2():
        output = torch.nn.functional.scaled_mm(
            input,
            weight,
            scale_a=scale_a,
            scale_recipe_a=ScalingType.TensorWise,
            scale_b=scale_b,
            scale_recipe_b=ScalingType.TensorWise,
            swizzle_a=SwizzleType.NO_SWIZZLE,
            swizzle_b=SwizzleType.NO_SWIZZLE,
            bias=bias,
            output_dtype=out_dtype,
        )
    else:
        output = torch._scaled_mm(
            input,
            weight,
            bias=bias,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=out_dtype,
        )

        # Handle tuple return in older versions
        if isinstance(output, tuple):
            output = output[0]

    # Manually apply scale_result if provided
    if scale_result is not None:
        output = output * scale_result.to(output.dtype)
        
    return output

def scaled_mm_blockwise(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_scale_a: torch.Tensor,
    tensor_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    tensor_scale_b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    swizzle_a: Optional['SwizzleType'] = [SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
    swizzle_b: Optional['SwizzleType'] = [SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
) -> torch.Tensor:
    if has_scaled_mm_v2():
        return torch.nn.functional.scaled_mm(
            input,
            weight,
            scale_a=[block_scale_a, tensor_scale_a],
            scale_recipe_a=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            scale_b=[block_scale_b, tensor_scale_b],
            scale_recipe_b=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            swizzle_a=swizzle_a,
            swizzle_b=swizzle_b,
            bias=bias,
            output_dtype=out_dtype,
            use_fast_accum=True
        )
    else:
        alpha = tensor_scale_a * tensor_scale_b
        output = torch._scaled_mm(
            input,
            weight,
            scale_a=block_scale_a.view(-1),
            scale_b=block_scale_b.view(-1),
            out_dtype=out_dtype,
        )
        
        # Handle tuple return
        if isinstance(output, tuple):
            output = output[0]
        output = output * alpha.to(output.dtype)
        if bias is not None:
            output = output + bias
        
        return output

# Version info for debugging
def get_pytorch_version_info() -> dict[str, str | bool]:
    """Get PyTorch version information for debugging.
    
    Returns:
        Dictionary with version info and feature flags
    """
    return {
        "torch_version": torch.__version__,
        "parsed_version": str(_TORCH_VERSION),
        "has_scaled_mm_v2": has_scaled_mm_v2(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
