__all__ = [
    "apply_rope",
    "apply_rope1",
    "dequantize_int8_blockwise",
    "dequantize_int8_weight",
    "dequantize_nvfp4",
    "dequantize_per_tensor_fp8",
    "int8_addmm",
    "int8_gemm",
    "quantize_int8_blockwise",
    "quantize_int8_weight",
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_per_tensor_fp8",
]

# Try to import triton and register if available
_TRITON_AVAILABLE = True
_TRITON_ERROR = None

try:
    import triton  # noqa: F401

    from .quantization import (
        dequantize_int8_blockwise,
        dequantize_int8_weight,
        dequantize_nvfp4,
        dequantize_per_tensor_fp8,
        int8_addmm,
        int8_gemm,
        quantize_int8_blockwise,
        quantize_int8_weight,
        quantize_mxfp8,
        quantize_nvfp4,
        quantize_per_tensor_fp8,
    )
    from .rope import apply_rope, apply_rope1
except ImportError as e:
    _TRITON_AVAILABLE = False
    _TRITON_ERROR = f"ImportError: {e!s}"


def _build_constraints() -> dict:
    import torch

    from comfy_kitchen.constraints import (
        ExactDims,
        FunctionConstraints,
        ParamConstraint,
    )

    cuda_devices = frozenset({"cuda"})
    standard_floats = frozenset({torch.float32, torch.float16, torch.bfloat16})

    return {
        "quantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "output_type": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2}),
                ),
            },
            default_devices=cuda_devices,
        ),
        "dequantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2}),
                ),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "output_type": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=cuda_devices,
        ),
        "quantize_nvfp4": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=standard_floats,
                    shape_rules=(ExactDims(2),),
                ),
                "per_tensor_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
            },
            default_devices=cuda_devices,
        ),
        "quantize_mxfp8": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=standard_floats,
                    shape_rules=(ExactDims(2),),
                ),
            },
            default_devices=cuda_devices,
        ),
        # Uses inline PTX: cvt.rn.f16x2.e2m1x2 (SM100/Blackwell instruction)
        "dequantize_nvfp4": FunctionConstraints(
            params={
                "qx": ParamConstraint(
                    dtypes=frozenset({torch.uint8}),
                    shape_rules=(ExactDims(2),),
                ),
                "per_tensor_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "block_scales": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn}),
                ),
                "output_type": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=cuda_devices,
            min_compute_capability=(10, 0),  # SM100 required for cvt.rn.f16x2.e2m1x2
        ),
        "apply_rope1": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=cuda_devices,
        ),
        "apply_rope": FunctionConstraints(
            params={
                "xq": ParamConstraint(dtypes=standard_floats),
                "xk": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=cuda_devices,
        ),
        # INT8 block-wise quantization
        "quantize_int8_blockwise": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=cuda_devices,
            min_compute_capability=(7, 5),  # INT8 tensor cores from Turing
        ),
        "dequantize_int8_blockwise": FunctionConstraints(
            params={
                "qx": ParamConstraint(dtypes=frozenset({torch.int8})),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
            },
            default_devices=cuda_devices,
            min_compute_capability=(7, 5),
        ),
        "quantize_int8_weight": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=standard_floats,
                    shape_rules=(ExactDims(2),),
                ),
            },
            default_devices=cuda_devices,
            min_compute_capability=(7, 5),
        ),
        "dequantize_int8_weight": FunctionConstraints(
            params={
                "qx": ParamConstraint(
                    dtypes=frozenset({torch.int8}),
                    shape_rules=(ExactDims(2),),
                ),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
            },
            default_devices=cuda_devices,
            min_compute_capability=(7, 5),
        ),
        "int8_gemm": FunctionConstraints(
            params={
                "a": ParamConstraint(dtypes=frozenset({torch.int8})),
                "a_s": ParamConstraint(dtypes=frozenset({torch.float32})),
                "b": ParamConstraint(
                    dtypes=frozenset({torch.int8}),
                    shape_rules=(ExactDims(2),),
                ),
                "b_s": ParamConstraint(
                    dtypes=frozenset({torch.float32}),
                    shape_rules=(ExactDims(2),),
                ),
            },
            default_devices=cuda_devices,
            min_compute_capability=(7, 5),
        ),
        "int8_addmm": FunctionConstraints(
            params={
                "a": ParamConstraint(dtypes=frozenset({torch.int8})),
                "a_s": ParamConstraint(dtypes=frozenset({torch.float32})),
                "b": ParamConstraint(
                    dtypes=frozenset({torch.int8}),
                    shape_rules=(ExactDims(2),),
                ),
                "b_s": ParamConstraint(
                    dtypes=frozenset({torch.float32}),
                    shape_rules=(ExactDims(2),),
                ),
            },
            default_devices=cuda_devices,
            min_compute_capability=(7, 5),
        ),
    }


def _register():
    import torch

    from comfy_kitchen.registry import registry

    if not _TRITON_AVAILABLE:
        registry.mark_unavailable("triton", _TRITON_ERROR)
        return

    if not torch.cuda.is_available():
        registry.mark_unavailable("triton", "CUDA not available on this system")
        return

    registry.register(
        name="triton",
        module=__import__(__name__, fromlist=__all__),
        capabilities=_build_constraints(),
    )


_register()

