"""INT8 block-wise quantized Linear layer implementation."""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from comfy_kitchen.tensor import BlockWiseINT8Layout, QuantizedTensor

logger = logging.getLogger(__name__)


class INT8Linear(nn.Module):
    """Linear layer with INT8 block-wise quantized weights.

    A memory-efficient linear layer that uses INT8 quantization with per-block
    scaling. Stores weights in INT8 format with 128x128 block-wise scales,
    reducing memory by ~50% compared to FP16/BF16.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: nn.Parameter | None = None,
        weight_qt: QuantizedTensor | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.weight = nn.Parameter(weight_qt, requires_grad=False) if weight_qt is not None else None
        self.bias = bias

    @classmethod
    def from_linear(cls, linear: nn.Linear, device: str = "cuda") -> INT8Linear:
        """Create an INT8Linear from a standard nn.Linear.

        Args:
            linear: Source linear layer.
            device: Target device for quantized weights.

        Returns:
            INT8Linear with quantized weights.
        """
        weight = linear.weight.data
        compute_dtype = weight.dtype

        # Move to target device if needed
        if weight.device.type != device.split(":")[0]:
            weight = weight.to(device)

        # Quantize weights using BlockWiseINT8Layout
        weight_qt = QuantizedTensor.from_float(weight, "BlockWiseINT8Layout", is_weight=True)

        bias = linear.bias
        if bias is not None and bias.device.type != device.split(":")[0]:
            bias = nn.Parameter(bias.data.to(device))

        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=bias,
            weight_qt=weight_qt,
            compute_dtype=compute_dtype,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with INT8 quantization.

        Quantizes input dynamically and uses INT8 GEMM when both operands
        are quantized. Falls back to dequantization if needed.
        """
        input_shape = x.shape
        tensor_3d = x.ndim == 3

        if not isinstance(x, QuantizedTensor):
            if tensor_3d:
                x = x.reshape(-1, input_shape[2])

            if x.ndim != 2:
                # Fallback for unsupported shapes
                return torch.nn.functional.linear(
                    x.reshape(input_shape),
                    self.weight.dequantize(),
                    self.bias
                )

            # Dynamically quantize input (1D block-wise for activations)
            x = QuantizedTensor.from_float(x, "BlockWiseINT8Layout", is_weight=False)

        # Dispatch via __torch_dispatch__ - uses INT8 GEMM if available
        output = torch.nn.functional.linear(x, self.weight, self.bias)

        if tensor_3d:
            output = output.reshape(input_shape[0], input_shape[1], self.out_features)

        return output

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        """Load INT8 quantized weights from state dict."""
        weight = state_dict.pop(prefix + "weight", None)
        weight_scale = state_dict.pop(prefix + "weight_scale", None)

        if weight is not None and weight_scale is not None:
            weight = weight.to(device=self.weight.device if self.weight is not None else "cuda")
            weight_scale = weight_scale.to(device=weight.device)

            params = BlockWiseINT8Layout.Params(
                scale=weight_scale,
                orig_dtype=self.compute_dtype,
                orig_shape=(self.out_features, self.in_features),
                block_size=128,
                is_weight=True,
            )

            self.weight = nn.Parameter(
                QuantizedTensor(weight, "BlockWiseINT8Layout", params),
                requires_grad=False
            )

            manually_loaded_keys = [prefix + "weight", prefix + "weight_scale"]
            for key in manually_loaded_keys:
                if key in missing_keys:
                    missing_keys.remove(key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                       strict, missing_keys, unexpected_keys, error_msgs)

    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return state dict with INT8 weight tensors."""
        state_dict = super().state_dict(*args, **kwargs)
        if "weight" in state_dict:
            del state_dict["weight"]
        if self.weight is not None:
            state_dict.update(self.weight.state_dict("weight"))
        return state_dict


def patch_model_with_int8(
    model: nn.Module,
    min_features: int = 128,  # INT8 needs 128-aligned dims
    skip_names: list[str] | None = None,
    verbose: bool = False,
) -> tuple[nn.Module, int]:
    """Patch all suitable Linear layers in a model to use INT8 quantization.

    Args:
        model: The model to patch.
        min_features: Minimum feature size (must be >= 128 for block alignment).
        skip_names: List of layer name patterns to skip.
        verbose: Print detailed progress.

    Returns:
        Tuple of (patched model, number of layers patched).
    """
    skip_names = skip_names or []
    patched_count = 0
    skipped_reasons: dict[str, list[str]] = {
        "skip_pattern": [],
        "too_small": [],
        "not_aligned": [],
        "failed": [],
    }

    def should_skip(name: str) -> bool:
        return any(skip in name for skip in skip_names)

    def is_aligned(n: int, block_size: int = 128) -> bool:
        return n % block_size == 0

    def patch_recursive(module: nn.Module, prefix: str = "") -> None:
        nonlocal patched_count

        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                if should_skip(full_name):
                    skipped_reasons["skip_pattern"].append(full_name)
                    continue

                if child.in_features < min_features or child.out_features < min_features:
                    skipped_reasons["too_small"].append(
                        f"{full_name} ({child.in_features}x{child.out_features})"
                    )
                    continue

                if not is_aligned(child.in_features) or not is_aligned(child.out_features):
                    skipped_reasons["not_aligned"].append(
                        f"{full_name} ({child.in_features}x{child.out_features})"
                    )
                    continue

                try:
                    int8_linear = INT8Linear.from_linear(child)
                    setattr(module, name, int8_linear)
                    patched_count += 1
                    if verbose:
                        print(f"    ✓ {full_name} ({child.in_features}x{child.out_features})")
                except Exception as e:
                    skipped_reasons["failed"].append(f"{full_name}: {e}")
            else:
                patch_recursive(child, full_name)

    patch_recursive(model)

    if verbose:
        if skipped_reasons["skip_pattern"]:
            print(f"  Skipped (pattern match): {len(skipped_reasons['skip_pattern'])}")
        if skipped_reasons["too_small"]:
            print(f"  Skipped (too small): {len(skipped_reasons['too_small'])}")
        if skipped_reasons["not_aligned"]:
            print(f"  Skipped (not 128-aligned): {len(skipped_reasons['not_aligned'])}")
        if skipped_reasons["failed"]:
            print(f"  Failed to patch: {len(skipped_reasons['failed'])}")
            for reason in skipped_reasons["failed"]:
                print(f"    ✗ {reason}")

    return model, patched_count


class ApplyINT8Quantization:
    """ComfyUI node that applies INT8 quantization to model linear layers."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "min_features": ("INT", {"default": 128, "min": 128, "max": 4096, "step": 128}),
                "skip_patterns": ("STRING", {"default": "", "multiline": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "comfy_kitchen/quantization"

    def apply(
        self,
        model: Any,
        min_features: int = 128,
        skip_patterns: str = "",
        debug: bool = False,
    ) -> tuple[Any]:
        skip_names = [s.strip() for s in skip_patterns.split("\n") if s.strip()]

        model_patcher = model.clone()
        diffusion_model = model_patcher.model.diffusion_model

        # Count layers before patching
        total_linear = sum(1 for m in diffusion_model.modules() if isinstance(m, nn.Linear))

        if debug:
            print("\nINT8 Patching:")
            print(f"  Total nn.Linear layers: {total_linear}")
            print(f"  Min features (128-aligned): {min_features}")
            if skip_names:
                print(f"  Skip patterns: {skip_names}")

        _, patched_count = patch_model_with_int8(
            diffusion_model,
            min_features=min_features,
            skip_names=skip_names,
            verbose=debug,
        )

        if debug:
            print(f"  Successfully patched: {patched_count}")
            print()

        logger.info(f"INT8: Patched {patched_count} linear layers")
        return (model_patcher,)


NODE_CLASS_MAPPINGS = {
    "ApplyINT8Quantization": ApplyINT8Quantization,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyINT8Quantization": "Apply INT8 Quantization",
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.small = nn.Linear(16, 16)  # Too small, will skip

        def forward(self, x):
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            x = torch.relu(x)
            x = self.fc3(x)
            return x

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available, skipping test")
        exit(0)

    print("=" * 80)
    print("INT8 Block-wise Quantization Test")
    print("=" * 80)

    model = SimpleModel().to(device).to(torch.bfloat16)
    print(f"Before patching: {model}")

    print("\nINT8 Patching:")
    model, count = patch_model_with_int8(model, min_features=128, verbose=True)
    print(f"  Total patched: {count}")
    print(f"\nAfter patching: {model}")

    print("\n" + "=" * 80)
    print("Running forward pass:")
    print("=" * 80)

    x = torch.randn(32, 256, device=device, dtype=torch.bfloat16)
    output = model(x)
    torch.cuda.synchronize()

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Forward pass successful!")
