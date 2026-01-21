# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

import triton
import triton.language as tl
from comfy_kitchen.float_utils import (
    F8_E4M3_MAX,
    F8_E5M2_MAX,
    ceil_div,
)


@triton.jit
def quantize_fp8_kernel_tl(
    x_ptr,
    output_ptr,
    scale_ptr,
    lp_max,
    n_elements,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements

    # Load scale value from device tensor
    scale = tl.load(scale_ptr)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scaled = x.to(tl.float32) / scale

    clamped = tl.maximum(tl.minimum(scaled, lp_max), -lp_max)
    tl.store(output_ptr + offsets, clamped, mask=mask)


def quantize_per_tensor_fp8(
    x: torch.Tensor, scale: torch.Tensor, output_type: torch.dtype = torch.float8_e4m3fn
) -> torch.Tensor:
    if output_type == torch.float8_e4m3fn:
        lp_max = F8_E4M3_MAX
    elif output_type == torch.float8_e5m2:
        lp_max = F8_E5M2_MAX
    else:
        raise ValueError(
            f"Unsupported output_type: {output_type}. Expected torch.float8_e4m3fn or torch.float8_e5m2"
        )

    if not x.is_contiguous():
        x = x.contiguous()

    orig_shape = x.shape
    x_flat = x.flatten()
    n_elements = x_flat.numel()

    output = torch.empty_like(x_flat, dtype=output_type)


    if n_elements < 32768:  # < 32K elements
        block_size = 128
    elif n_elements < 131072:  # < 128K elements
        block_size = 256
    elif n_elements < 524288:  # < 512K elements
        block_size = 512
    else:
        block_size = 1024

    grid = (triton.cdiv(n_elements, block_size),)

    quantize_fp8_kernel_tl[grid](
        x_flat,
        output,
        scale,
        lp_max,
        n_elements,
        block_size=block_size,
    )

    output = output.view(orig_shape)

    return output


@triton.jit
def dequantize_fp8_kernel_tl(
    x_ptr,
    output_ptr,
    scale_ptr,
    n_elements,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements

    # Load scale value from device tensor
    scale = tl.load(scale_ptr)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    dequantized = x.to(tl.float32) * scale

    tl.store(output_ptr + offsets, dequantized, mask=mask)


def dequantize_per_tensor_fp8(
    x: torch.Tensor, scale: torch.Tensor, output_type: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()

    orig_shape = x.shape
    x_flat = x.flatten()
    n_elements = x_flat.numel()

    output = torch.empty_like(x_flat, dtype=output_type)

    if n_elements < 32768:  # < 32K elements
        block_size = 128
    elif n_elements < 131072:  # < 128K elements
        block_size = 256
    elif n_elements < 524288:  # < 512K elements
        block_size = 512
    else:
        block_size = 1024

    grid = (triton.cdiv(n_elements, block_size),)

    dequantize_fp8_kernel_tl[grid](
        x_flat,
        output,
        scale,
        n_elements,
        block_size=block_size,
    )

    output = output.view(orig_shape)

    return output


@triton.jit
def _compute_swizzled_scale_offset(
    in_row,
    in_col,
    n_col_blocks,
    padded_scale_cols,
):
    """Compute the swizzled offset for a scale at logical position (in_row, in_col).

    This implements the cuBLAS blocked layout transformation (to_blocked).
    Used by both quantize (write) and dequantize (read) kernels.
    """
    # Compute which 128x4 block this element belongs to
    row_block = in_row // 128
    col_block = in_col // 4

    # Position within the 128x4 block
    in_block_row = in_row % 128
    in_block_col = in_col % 4

    # Map through the swizzle transformations
    sub_block = in_block_row // 32
    fine_row = in_block_row % 32

    combined_block = row_block * n_col_blocks + col_block
    intermediate_col = sub_block * 4 + in_block_col

    # Flatten intermediate and compute linear index
    linear_idx = combined_block * 512 + fine_row * 16 + intermediate_col

    # Convert to final 2D position and compute offset
    out_row = linear_idx // padded_scale_cols
    out_col = linear_idx % padded_scale_cols
    return out_row * padded_scale_cols + out_col


@triton.jit
def quantize_nvfp4_kernel_tl(
    x_ptr,
    packed_output_ptr,
    swizzled_scales_ptr,
    per_tensor_scale_ptr,
    m,
    n,
    num_blocks,
    scale_rows,
    scale_cols,
    padded_scale_rows,
    padded_scale_cols,
    block_size: tl.constexpr,
    blocks_per_program: tl.constexpr,
):
    """Single Triton kernel for NVFP4 quantization with packing and scale swizzling.

    Performs all operations in one kernel:
    1. Computes block-wise scales
    2. Quantizes and packs data to FP4
    3. Applies to_blocked swizzle pattern to scales

    Optimized with:
    - Vectorized processing of multiple blocks per thread
    - Efficient packing using interleave operations
    - Coalesced memory accesses

    Args:
        x_ptr: Input tensor pointer (m x n)
        packed_output_ptr: Output packed FP4 data (m x n//2)
        swizzled_scales_ptr: Output swizzled FP8 block scales (padded_scale_rows x padded_scale_cols)
        per_tensor_scale_ptr: Pointer to global scaling factor tensor
        m: Number of rows in input
        n: Number of columns in input (must be divisible by block_size)
        num_blocks: Number of blocks per row (n // block_size)
        scale_rows: Unpadded scale rows (m)
        scale_cols: Unpadded scale cols (num_blocks)
        padded_scale_rows: Padded scale rows for swizzle
        padded_scale_cols: Padded scale cols for swizzle
        block_size: Size of each quantization block (typically 16)
        blocks_per_program: Number of blocks to process per program
    """
    # Get program IDs - each program processes blocks_per_program blocks
    pid_m = tl.program_id(axis=0)
    pid_n_base = tl.program_id(axis=1) * blocks_per_program

    # Load per-tensor scale value from device tensor
    per_tensor_scale = tl.load(per_tensor_scale_ptr)

    # Process multiple blocks per program
    for block_offset in range(blocks_per_program):
        pid_n = pid_n_base + block_offset

        # Skip if beyond num_blocks
        if pid_n < num_blocks:
            # Calculate offsets for the input data block
            offs_n = pid_n * block_size + tl.arange(0, block_size)
            mask = offs_n < n
            x_offs = pid_m * n + offs_n

            # Load input data block
            x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

            # Compute block-wise absolute maximum
            x_abs = tl.abs(x)
            max_abs = tl.max(x_abs, axis=0)

            # Calculate block scale: block_scale = max_abs / F4_E2M1_MAX (6.0)
            block_scale = max_abs / 6.0

            # Scale block scale to FP8
            scaled_block_scale = block_scale / per_tensor_scale
            scaled_block_scale = tl.minimum(scaled_block_scale, 448.0)

            # Round to FP8 precision
            scaled_block_scale_fp8 = scaled_block_scale.to(tl.float8e4nv)

            # Compute swizzled position and store scale
            n_col_blocks = tl.cdiv(scale_cols, 4)
            swizzled_offs = _compute_swizzled_scale_offset(
                pid_m, pid_n, n_col_blocks, padded_scale_cols
            )
            if pid_m < scale_rows and pid_n < scale_cols:
                tl.store(swizzled_scales_ptr + swizzled_offs, scaled_block_scale_fp8)

            # Calculate total scale for data quantization
            scaled_block_scale_fp32 = scaled_block_scale_fp8.to(tl.float32)
            total_scale = per_tensor_scale * scaled_block_scale_fp32
            zero_scale_mask = total_scale < 1e-10
            total_scale = tl.where(zero_scale_mask, 1.0, total_scale)

            # Scale data (satfinite modifier in PTX will handle clamping)
            data_scaled = x / total_scale
            data_scaled = tl.where(zero_scale_mask, 0.0, data_scaled)

            # We want to pack: (v0,v1), (v2,v3), ..., (v14,v15)
            pair_idx = tl.arange(0, block_size // 2)
            even_idx = pair_idx * 2
            odd_idx = pair_idx * 2 + 1

            # Extract even and odd elements using one-hot selection
            indices = tl.arange(0, block_size)
            f32_even = tl.sum(tl.where(indices == even_idx[:, None], data_scaled, 0), axis=1)
            f32_odd = tl.sum(tl.where(indices == odd_idx[:, None], data_scaled, 0), axis=1)

            packed_bytes_u16 = tl.inline_asm_elementwise(
                asm="""
                {
                    .reg .b8 fp4_byte;
                    .reg .b16 result;
                    cvt.rn.satfinite.e2m1x2.f32 fp4_byte, $1, $2;
                    mov.b16 result, {fp4_byte, 0};
                    mov.u16 $0, result;
                }
                """,
                constraints="=h,f,f",
                args=[f32_even, f32_odd],
                dtype=tl.uint16,
                is_pure=True,
                pack=1,
            )
            # Extract the low byte
            packed_bytes = (packed_bytes_u16 & 0xFF).to(tl.uint8)

            # Store packed bytes
            out_offs = pid_m * (n // 2) + pid_n * (block_size // 2) + pair_idx
            out_mask = (pid_n * block_size + even_idx) < n
            tl.store(packed_output_ptr + out_offs, packed_bytes, mask=out_mask)


def quantize_nvfp4(
    x: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    epsilon: float = 0.0,
    pad_16x: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Note: epsilon is accepted for API compatibility but not currently used
    orig_shape = x.shape

    # Handle padding
    if pad_16x:
        rows, cols = x.shape
        pad_rows = (rows + 15) // 16 * 16 - rows
        pad_cols = (cols + 15) // 16 * 16 - cols
        if pad_rows > 0 or pad_cols > 0:
            x = torch.nn.functional.pad(x, (0, pad_cols, 0, pad_rows))
            # Note: We update orig_shape because the output tensor logic below assumes x.shape matches
            # what we want to produce. If we pad here, we want the padded output.
            orig_shape = x.shape

    block_size = 16

    # Reshape for block processing
    x = x.reshape(orig_shape[0], -1, block_size)
    m, num_blocks, _ = x.shape
    n = num_blocks * block_size

    # Flatten to 2D for kernel processing
    x_2d = x.reshape(m, n).contiguous()

    # Calculate swizzled scale dimensions
    scale_rows = m
    scale_cols = num_blocks
    n_row_blocks = ceil_div(scale_rows, 128)
    n_col_blocks = ceil_div(scale_cols, 4)
    padded_scale_rows = n_row_blocks * 128
    padded_scale_cols = n_col_blocks * 4

    # Allocate output tensors
    packed_output = torch.empty((m, n // 2), dtype=torch.uint8, device=x.device)
    # Use zeros for scales to avoid garbage in padded regions
    swizzled_scales = torch.zeros(
        (padded_scale_rows, padded_scale_cols),
        dtype=torch.float8_e4m3fn,
        device=x.device
    )

    # Determine blocks per program based on tensor size for better occupancy
    total_blocks = m * num_blocks
    if total_blocks < 1024:
        blocks_per_program = 1
    elif total_blocks < 4096:
        blocks_per_program = 2
    else:
        blocks_per_program = 4

    # Launch single kernel that does everything
    grid = (m, triton.cdiv(num_blocks, blocks_per_program))

    quantize_nvfp4_kernel_tl[grid](
        x_2d,
        packed_output,
        swizzled_scales,
        per_tensor_scale,
        m,
        n,
        num_blocks,
        scale_rows,
        scale_cols,
        padded_scale_rows,
        padded_scale_cols,
        block_size=block_size,
        blocks_per_program=blocks_per_program,
    )

    # Reshape packed output to original shape (with last dim halved)
    packed_shape = list(orig_shape)
    packed_shape[-1] = packed_shape[-1] // 2
    packed_output = packed_output.reshape(packed_shape)

    return packed_output, swizzled_scales


@triton.jit
def dequantize_nvfp4_kernel_tl(
    packed_ptr,
    scale_ptr,
    global_scale_ptr,
    output_ptr,
    n,
    scale_cols,
    n_col_blocks,
    padded_scale_cols,
    block_size: tl.constexpr,
    tile_size: tl.constexpr,
):
    """Dequantizes FP4 packed data using per-block scaling factors.

    Args:
        packed_ptr (tl.pointer): Pointer to packed uint8 tensor (m x n//2)
        scale_ptr (tl.pointer): Pointer to swizzled per-block scale tensor
        output_ptr (tl.pointer): Pointer to output tensor (m x n)
        global_scale_ptr (tl.pointer): Pointer to global scale tensor
        n (int): Number of columns in unpacked tensor
        scale_cols (int): Number of scale columns (n // block_size)
        n_col_blocks (int): Number of 4-column blocks in scales
        padded_scale_cols (int): Padded scale columns (n_col_blocks * 4)
        block_size (tl.constexpr): Size of each FP4 quantization block
        tile_size (tl.constexpr): Size of the processing tile (in packed elements)
    """
    # Get program ID for processing packed elements
    pid = tl.program_id(0)

    # Calculate packed element offsets (each packed element contains 2 FP4 values)
    packed_start = pid * tile_size
    packed_offs = packed_start + tl.arange(0, tile_size)

    # Calculate 2D coordinates for packed data
    packed_row_idx = packed_offs // (n // 2)
    packed_col_idx = packed_offs % (n // 2)

    # Create mask for packed data bounds checking
    packed_mask = packed_col_idx < (n // 2)

    # Load global scale
    global_scale = tl.load(global_scale_ptr)

    # Load packed data
    packed_data = tl.load(packed_ptr + packed_offs, mask=packed_mask, other=0)

    # Unpack packed FP4 values (uint8) to float16x2
    x_f16x2_packed = tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b8 byte0, byte1, byte2, byte3;
            mov.b32 {byte0, byte1, byte2, byte3}, $4;
            cvt.rn.f16x2.e2m1x2 $0, byte0;
            cvt.rn.f16x2.e2m1x2 $1, byte1;
            cvt.rn.f16x2.e2m1x2 $2, byte2;
            cvt.rn.f16x2.e2m1x2 $3, byte3;
        }
        """,
        constraints="=r,=r,=r,=r,r",
        args=[packed_data],
        dtype=tl.uint32,
        is_pure=True,
        pack=4,
    )
    val_low = (
        (x_f16x2_packed & 0xFFFF).cast(tl.uint16).cast(tl.float16, bitcast=True).cast(tl.float32)
    )
    val_high = (
        (x_f16x2_packed >> 16).cast(tl.uint16).cast(tl.float16, bitcast=True).cast(tl.float32)
    )

    # Calculate output positions for both values
    out_col_low = packed_col_idx * 2
    out_col_high = packed_col_idx * 2 + 1
    out_offs_low = packed_row_idx * n + out_col_low
    out_offs_high = packed_row_idx * n + out_col_high

    # Calculate block indices for scaling (logical positions in scale tensor)
    block_col_low = out_col_low // block_size
    block_col_high = out_col_high // block_size

    # Compute swizzled offsets for scale lookups
    scale_offs_low = _compute_swizzled_scale_offset(
        packed_row_idx, block_col_low, n_col_blocks, padded_scale_cols
    )
    scale_offs_high = _compute_swizzled_scale_offset(
        packed_row_idx, block_col_high, n_col_blocks, padded_scale_cols
    )

    # Load scaling factors from swizzled positions
    scale_low = tl.load(
        scale_ptr + scale_offs_low,
        mask=packed_mask & (block_col_low < scale_cols),
        other=1.0,
    )
    scale_high = tl.load(
        scale_ptr + scale_offs_high,
        mask=packed_mask & (block_col_high < scale_cols),
        other=1.0,
    )

    # Apply scaling
    # Note: Due to packing order ((even << 4) | odd) and PTX cvt.rn.f16x2.e2m1x2:
    # - val_low (from low nibble) contains odd-indexed values
    # - val_high (from high nibble) contains even-indexed values
    result_even = val_high * scale_low.to(tl.float32) * global_scale
    result_odd = val_low * scale_high.to(tl.float32) * global_scale

    # Store results
    out_mask_low = packed_mask & (out_col_low < n)
    out_mask_high = packed_mask & (out_col_high < n)

    tl.store(output_ptr + out_offs_low, result_even, mask=out_mask_low)
    tl.store(output_ptr + out_offs_high, result_odd, mask=out_mask_high)


def dequantize_nvfp4(
    qx: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scales: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    # Triton backend: fused kernel with inline SM100 cvt.rn.f16x2.e2m1x2 instruction
    block_size = 16
    tile_size = 128

    packed_n = qx.shape[-1]
    n = packed_n * 2
    scale_cols = n // block_size

    # Compute swizzle layout parameters
    n_col_blocks = ceil_div(scale_cols, 4)
    padded_scale_cols = n_col_blocks * 4

    # Create output tensor with proper shape handling
    output_shape = list(qx.shape)
    output_shape[-1] = n
    output = torch.empty(output_shape, dtype=output_type, device=qx.device)

    # Calculate total number of elements and grid size
    def grid(meta):
        return (triton.cdiv(qx.numel(), meta["tile_size"]),)

    dequantize_nvfp4_kernel_tl[grid](
        qx,
        block_scales,
        per_tensor_scale,
        output,
        n,
        scale_cols,
        n_col_blocks,
        padded_scale_cols,
        block_size=block_size,
        tile_size=tile_size,
    )

    return output

@triton.jit
def quantize_mxfp8_kernel_tl(
    x_ptr,
    output_ptr,
    swizzled_scales_ptr,
    m,
    n,
    num_blocks,
    scale_rows,
    scale_cols,
    padded_scale_cols,
    block_size: tl.constexpr,
    blocks_per_program: tl.constexpr,
):
    """Single Triton kernel for MXFP8 quantization with E8M0 scale swizzling.

    Performs:
    1. Computes block-wise max and converts to E8M0 (power-of-2) scales
    2. Quantizes data to FP8 E4M3
    3. Applies to_blocked swizzle pattern to E8M0 scales

    Args:
        x_ptr: Input tensor pointer (m x n)
        output_ptr: Output FP8 data (m x n)
        swizzled_scales_ptr: Output swizzled E8M0 block scales
        m: Number of rows in input
        n: Number of columns in input (must be divisible by block_size)
        num_blocks: Number of blocks per row (n // block_size)
        scale_rows: Unpadded scale rows (m)
        scale_cols: Unpadded scale cols (num_blocks)
        padded_scale_cols: Padded scale cols for swizzle
        block_size: Size of each quantization block (32 for MXFP8)
        blocks_per_program: Number of blocks to process per program
    """
    # Get program IDs
    pid_m = tl.program_id(axis=0)
    pid_n_base = tl.program_id(axis=1) * blocks_per_program

    # FP8 E4M3 max value
    fp8_max: tl.constexpr = 448.0

    # Process multiple blocks per program
    for block_offset in range(blocks_per_program):
        pid_n = pid_n_base + block_offset

        if pid_n < num_blocks:
            # Calculate offsets for the input data block
            offs_n = pid_n * block_size + tl.arange(0, block_size)
            mask = offs_n < n
            x_offs = pid_m * n + offs_n

            # Load input data block
            x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

            # Compute block-wise absolute maximum
            x_abs = tl.abs(x)
            max_abs = tl.max(x_abs, axis=0)

            # Compute E8M0 scale: find power-of-2 that covers max_abs
            # E8M0 has bias 127, so scale = 2^(exp - 127)
            # We want 2^exp >= max_abs / fp8_max, so exp = ceil(log2(max_abs / fp8_max)) + 127
            # Using floor(log2(x)) + 1 for ceiling
            scale_ratio = max_abs / fp8_max
            # Clamp to avoid log2(0) and ensure valid E8M0 range
            scale_ratio = tl.maximum(scale_ratio, 2.0 ** (-127))  # min E8M0 value
            scale_ratio = tl.minimum(scale_ratio, 2.0 ** 127)     # max E8M0 value

            # Compute exponent: round up to next power of 2
            log2_ratio = tl.log2(scale_ratio)
            exp_unbiased = tl.math.ceil(log2_ratio).to(tl.int32)
            exp_biased = exp_unbiased + 127  # E8M0 bias

            # Clamp to valid E8M0 range [0, 254] (255 is NaN)
            exp_biased = tl.maximum(exp_biased, 0)
            exp_biased = tl.minimum(exp_biased, 254)

            # Compute actual scale value for quantization
            block_scale = tl.exp2((exp_biased - 127).to(tl.float32))

            # Store E8M0 scale in swizzled layout
            n_col_blocks = tl.cdiv(scale_cols, 4)
            swizzled_offs = _compute_swizzled_scale_offset(
                pid_m, pid_n, n_col_blocks, padded_scale_cols
            )
            if pid_m < scale_rows and pid_n < scale_cols:
                # Store as uint8 (E8M0 is just an 8-bit exponent)
                tl.store(swizzled_scales_ptr + swizzled_offs, exp_biased.to(tl.uint8))

            # Quantize data to FP8
            # Handle zero scale to avoid division by zero
            safe_scale = tl.where(block_scale < 1e-30, 1.0, block_scale)
            data_scaled = x / safe_scale
            data_scaled = tl.where(block_scale < 1e-30, 0.0, data_scaled)

            # Clamp to FP8 range and convert
            data_clamped = tl.maximum(tl.minimum(data_scaled, fp8_max), -fp8_max)

            # Store as FP8 E4M3
            out_offs = pid_m * n + offs_n
            tl.store(output_ptr + out_offs, data_clamped.to(tl.float8e4nv), mask=mask)


def quantize_mxfp8(
    x: torch.Tensor,
    pad_32x: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to MXFP8 format with block-wise E8M0 scaling.

    MXFP8 uses block size 32 with power-of-2 (E8M0) block scales.

    Args:
        x: Input tensor (2D, shape M x K)
        pad_32x: If True, pad dimensions to be divisible by 32

    Returns:
        Tuple of (quantized_fp8_tensor, block_scales_e8m0)
    """
    block_size = 32

    # Handle padding
    if pad_32x:
        rows, cols = x.shape
        pad_rows = (rows + 31) // 32 * 32 - rows
        pad_cols = (cols + 31) // 32 * 32 - cols
        if pad_rows > 0 or pad_cols > 0:
            x = torch.nn.functional.pad(x, (0, pad_cols, 0, pad_rows))

    m, n = x.shape
    num_blocks = n // block_size

    # Ensure contiguous
    x = x.contiguous()

    # Calculate swizzled scale dimensions
    scale_rows = m
    scale_cols = num_blocks
    n_row_blocks = ceil_div(scale_rows, 128)
    n_col_blocks = ceil_div(scale_cols, 4)
    padded_scale_rows = n_row_blocks * 128
    padded_scale_cols = n_col_blocks * 4

    # Allocate output tensors
    output = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    # Use zeros for scales to avoid garbage in padded regions
    swizzled_scales = torch.zeros(
        (padded_scale_rows, padded_scale_cols),
        dtype=torch.uint8,  # E8M0 stored as uint8
        device=x.device
    )

    # Determine blocks per program
    total_blocks = m * num_blocks
    if total_blocks < 1024:
        blocks_per_program = 1
    elif total_blocks < 4096:
        blocks_per_program = 2
    else:
        blocks_per_program = 4

    # Launch kernel
    grid = (m, triton.cdiv(num_blocks, blocks_per_program))

    quantize_mxfp8_kernel_tl[grid](
        x,
        output,
        swizzled_scales,
        m,
        n,
        num_blocks,
        scale_rows,
        scale_cols,
        padded_scale_cols,
        block_size=block_size,
        blocks_per_program=blocks_per_program,
    )

    # Convert uint8 scales to float8_e8m0fnu
    swizzled_scales = swizzled_scales.view(torch.float8_e8m0fnu)

    return output, swizzled_scales


# =============================================================================
# INT8 Block-wise Quantization
# =============================================================================

@triton.jit
def int8_act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """Quantizes activation tensor block-wise along the last dimension.

    Args:
        x_ptr: Pointer to the input tensor.
        y_ptr: Pointer to the output quantized tensor.
        s_ptr: Pointer to the output scaling factors.
        BLOCK_SIZE: The size of each quantization block.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    amax = tl.max(tl.abs(x))
    s = amax / 127.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)




@triton.jit
def int8_act_dequant_kernel(x_ptr, s_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    """Dequantizes activation tensor using block-wise scaling."""
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.load(s_ptr + pid)
    y = x * s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)


@triton.jit
def int8_weight_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """Quantizes 2D weight tensor with block-wise scaling."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    amax = tl.max(tl.abs(x))
    s = amax / 127.0

    y = x / s
    y = y.to(y_ptr.dtype.element_ty)

    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n + pid_n, s)


@triton.jit
def int8_weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """Dequantizes 2D weight tensor using block-wise scaling."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def int8_gemm_kernel(
    a_ptr, b_ptr, c_ptr, a_s_ptr, b_s_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """INT8 matrix multiplication kernel with per-block scaling.

    Computes: C = (A @ B^T) with INT8 inputs and per-block scaling.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k

    k_blocks = k
    b_s_base = b_s_ptr + pid_n * k_blocks

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k_blocks):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_base + i)

        dot_prod = tl.dot(a, b, out_dtype=tl.int32)
        accumulator += dot_prod.to(tl.float32) * a_s[:, None] * b_s

        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1

    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


@triton.jit
def int8_gemm_addmm_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr, a_s_ptr, b_s_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """INT8 matrix multiplication with fused bias addition.

    Computes: C = (A @ B^T) + bias
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k

    k_blocks = k
    b_s_base = b_s_ptr + pid_n * k_blocks

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k_blocks):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_base + i)

        dot_prod = tl.dot(a, b, out_dtype=tl.int32)
        accumulator += dot_prod.to(tl.float32) * a_s[:, None] * b_s

        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1

    # Add bias if provided
    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_n[None, :]
        bias = tl.load(bias_ptrs, mask=offs_n[None, :] < N, other=0.0)
        accumulator += bias

    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def quantize_int8(
    x: torch.Tensor,
    block_size: int = 128,
    is_weight: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise INT8 quantization.

    Args:
        x: Input tensor. For weights (is_weight=True), must be 2D with both
           dims divisible by block_size. For activations, last dim must be
           divisible by block_size.
        block_size: Quantization block size (default 128).
        is_weight: If True, use 2D blocking for weights. If False, use 1D
                   blocking along last dimension for activations.

    Returns:
        Tuple of (qdata, scale):
            - qdata: Quantized INT8 tensor with same shape as input
            - scale: Per-block scaling factors
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"

    if is_weight:
        # 2D block-wise quantization for weights
        assert x.dim() == 2, f"Weight must be 2D, got {x.dim()}D"
        M, N = x.size()
        assert M % block_size == 0 and N % block_size == 0, (
            f"Dimensions must be divisible by block_size={block_size}, got shape {x.shape}"
        )

        y = torch.empty_like(x, dtype=torch.int8)
        s = x.new_empty(M // block_size, N // block_size, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_SIZE"]),
            triton.cdiv(N, meta["BLOCK_SIZE"]),
        )
        int8_weight_quant_kernel[grid](x, y, s, M, N, BLOCK_SIZE=block_size)
    else:
        # 1D block-wise quantization for activations
        assert x.size(-1) % block_size == 0, (
            f"Last dimension {x.size(-1)} must be divisible by block_size {block_size}"
        )

        y = torch.empty_like(x, dtype=torch.int8)
        s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)

        num_programs = s.numel()
        grid = lambda meta: (num_programs,)
        int8_act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)

    return y, s


def dequantize_int8(
    qx: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Block-wise INT8 dequantization.

    Automatically detects activation vs weight based on scale tensor shape:
    - 2D scale = weight (2D blocking)
    - Other = activation (1D blocking along last dim)

    Args:
        qx: Quantized INT8 tensor.
        scale: Per-block scaling factors.
        block_size: Block size used for quantization.
        output_dtype: Target output dtype.

    Returns:
        Dequantized tensor with original shape.
    """
    assert qx.is_contiguous() and scale.is_contiguous(), "Tensors must be contiguous"
    is_weight = (scale.dim() == 2 and qx.dim() == 2)

    if is_weight:
        # 2D block-wise dequantization for weights
        M, N = qx.size()
        y = torch.empty_like(qx, dtype=output_dtype)

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_SIZE"]),
            triton.cdiv(N, meta["BLOCK_SIZE"]),
        )
        int8_weight_dequant_kernel[grid](qx, scale, y, M, N, BLOCK_SIZE=block_size)
    else:
        # 1D block-wise dequantization for activations
        assert qx.size(-1) % block_size == 0, (
            f"Last dimension {qx.size(-1)} must be divisible by block_size {block_size}"
        )

        y = torch.empty_like(qx, dtype=output_dtype)
        num_programs = scale.numel()
        grid = lambda meta: (num_programs,)
        int8_act_dequant_kernel[grid](qx, scale, y, BLOCK_SIZE=block_size)

    return y


def scaled_mm_int8(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """INT8 matrix multiplication with block-wise scaling.

    Computes: C = A @ B^T + bias (linear semantics)

    Args:
        a: INT8 activations [..., K].
        b: INT8 weights [N, K].
        scale_a: Activation scales [..., K//block_size].
        scale_b: Weight scales [N//block_size, K//block_size].
        bias: Optional bias vector [N].
        out_dtype: Output dtype.

    Returns:
        Result tensor [..., N].
    """
    assert a.is_contiguous() and b.is_contiguous()
    assert scale_a.is_contiguous() and scale_b.is_contiguous()
    assert b.dim() == 2

    K = a.size(-1)
    M = a.numel() // K
    N = b.shape[0]

    assert b.size(1) == K

    c = a.new_empty(*a.size()[:-1], N, dtype=out_dtype)

    has_bias = bias is not None
    if has_bias:
        assert bias.is_contiguous()
        assert bias.dim() == 1 and bias.size(0) == N
        bias_ptr = bias
    else:
        bias_ptr = c  # Dummy pointer

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    int8_gemm_addmm_kernel[grid](
        a, b, c, bias_ptr, scale_a, scale_b, M, N, K,
        HAS_BIAS=has_bias,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128
    )
    return c


# =============================================================================
# Fused INT8 GEMM + Output Quantization
# =============================================================================
# These kernels fuse matmul with block-wise output quantization, avoiding
# materalization of the full-precision intermediate result. This is crucial
# for memory-bound models where INT8 layers feed into each other.


@triton.heuristics({
    "NUM_BLOCKS": lambda args: args["BLOCK_SIZE_N"] // args["out_block_size"],
})
@triton.jit
def int8_gemm_quant_kernel(
    a_ptr, b_ptr, c_ptr, c_s_ptr, a_s_ptr, b_s_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    out_block_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    """Fused INT8 matrix multiplication with output quantization.

    Computes: C_int8, C_scale = quantize(A @ B^T)

    This kernel fuses matmul and block-wise quantization in a single pass,
    avoiding materialization of the full-precision intermediate result.

    Args:
        a_ptr: Pointer to INT8 activations [..., K].
        b_ptr: Pointer to INT8 weights [N, K].
        c_ptr: Pointer to INT8 output [..., N].
        c_s_ptr: Pointer to output scales (shape: M x N/out_block_size).
        a_s_ptr: Pointer to activation scales.
        b_s_ptr: Pointer to weight scales.
        M: Number of rows in A and C.
        N: Number of columns in B and C.
        K: Inner dimension.
        out_block_size: Block size for output quantization.
        BLOCK_SIZE_M/N/K: Tile sizes for matmul.
        NUM_BLOCKS: BLOCK_SIZE_N // out_block_size (computed by heuristic).
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k

    k_blocks = k
    b_s_base = b_s_ptr + pid_n * k_blocks

    # Accumulate matmul result
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k_blocks):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_base + i)
        dot_prod = tl.dot(a, b, out_dtype=tl.int32)
        accumulator += dot_prod.to(tl.float32) * a_s[:, None] * b_s
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1

    # Quantize output: reshape to blocks and compute per-block scales
    accumulator_reshaped = tl.reshape(accumulator, (BLOCK_SIZE_M, NUM_BLOCKS, out_block_size))

    # Compute max per block: reduce over out_block_size dimension
    block_max = tl.max(tl.abs(accumulator_reshaped), axis=2)
    block_scale = tl.maximum(block_max / 127.0, 1e-8)

    # Quantize
    block_scale_broadcast = tl.reshape(block_scale, (BLOCK_SIZE_M, NUM_BLOCKS, 1))
    quantized = accumulator_reshaped / block_scale_broadcast
    quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0)
    quantized_int8 = quantized.to(c_ptr.dtype.element_ty)
    quantized_int8 = tl.reshape(quantized_int8, (BLOCK_SIZE_M, BLOCK_SIZE_N))

    # Store quantized output
    offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_actual = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m_actual[:, None] < M) & (offs_n_actual[None, :] < N)
    c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n_actual[None, :]
    tl.store(c_ptrs, quantized_int8, mask=mask)

    # Store scales in activation format: (M, N//out_block_size)
    n_scale_stride = N // out_block_size
    offs_m_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_scale = pid_n * NUM_BLOCKS + tl.arange(0, NUM_BLOCKS)
    scale_ptrs = c_s_ptr + offs_m_scale[:, None] * n_scale_stride + offs_n_scale[None, :]
    scale_mask = (offs_m_scale[:, None] < M) & (offs_n_scale[None, :] < n_scale_stride)
    tl.store(scale_ptrs, block_scale, mask=scale_mask)


@triton.heuristics({
    "NUM_BLOCKS": lambda args: args["BLOCK_SIZE_N"] // args["out_block_size"],
})
@triton.jit
def int8_gemm_addmm_quant_kernel(
    a_ptr, b_ptr, c_ptr, c_s_ptr, bias_ptr, a_s_ptr, b_s_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    out_block_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Fused INT8 GEMM with bias addition and output quantization.

    Computes: C_int8, C_scale = quantize(A @ B^T + bias)
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k

    k_blocks = k
    b_s_base = b_s_ptr + pid_n * k_blocks

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k_blocks):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_base + i)
        dot_prod = tl.dot(a, b, out_dtype=tl.int32)
        accumulator += dot_prod.to(tl.float32) * a_s[:, None] * b_s
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1

    # Add bias if provided
    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_n[None, :]
        bias = tl.load(bias_ptrs, mask=offs_n[None, :] < N, other=0.0)
        accumulator += bias

    # Quantize output
    accumulator_reshaped = tl.reshape(accumulator, (BLOCK_SIZE_M, NUM_BLOCKS, out_block_size))
    block_max = tl.max(tl.abs(accumulator_reshaped), axis=2)
    block_scale = tl.maximum(block_max / 127.0, 1e-8)

    block_scale_broadcast = tl.reshape(block_scale, (BLOCK_SIZE_M, NUM_BLOCKS, 1))
    quantized = accumulator_reshaped / block_scale_broadcast
    quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0)
    quantized_int8 = quantized.to(c_ptr.dtype.element_ty)
    quantized_int8 = tl.reshape(quantized_int8, (BLOCK_SIZE_M, BLOCK_SIZE_N))

    # Store quantized output
    offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_actual = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m_actual[:, None] < M) & (offs_n_actual[None, :] < N)
    c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n_actual[None, :]
    tl.store(c_ptrs, quantized_int8, mask=mask)

    # Store scales
    n_scale_stride = N // out_block_size
    offs_m_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_scale = pid_n * NUM_BLOCKS + tl.arange(0, NUM_BLOCKS)
    scale_ptrs = c_s_ptr + offs_m_scale[:, None] * n_scale_stride + offs_n_scale[None, :]
    scale_mask = (offs_m_scale[:, None] < M) & (offs_n_scale[None, :] < n_scale_stride)
    tl.store(scale_ptrs, block_scale, mask=scale_mask)


def scaled_mm_int8_quant(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused INT8 GEMM with output quantization.

    Computes: C_int8, C_scale = quantize(A @ B^T + bias)

    This avoids materializing the full-precision intermediate result,
    reducing memory bandwidth for INT8-to-INT8 layer chains.

    Args:
        a: INT8 activations [..., K].
        b: INT8 weights [N, K].
        scale_a: Activation scales [..., K//block_size].
        scale_b: Weight scales [N//block_size, K//block_size].
        bias: Optional bias vector [N].
        out_block_size: Block size for output quantization (default 128).

    Returns:
        Tuple of (quantized_output_int8, output_scales).
    """
    assert a.is_contiguous() and b.is_contiguous()
    assert scale_a.is_contiguous() and scale_b.is_contiguous()
    assert b.dim() == 2

    K = a.size(-1)
    M = a.numel() // K
    N = b.shape[0]
    batch_shape = a.size()[:-1]

    assert b.size(1) == K
    assert N % out_block_size == 0, f"N={N} must be divisible by out_block_size={out_block_size}"

    # Allocate output tensors
    c = a.new_empty(*batch_shape, N, dtype=torch.int8)
    n_blocks = N // out_block_size
    c_s = a.new_empty(M, n_blocks, dtype=torch.float32)

    has_bias = bias is not None
    if has_bias:
        assert bias.is_contiguous()
        assert bias.dim() == 1 and bias.size(0) == N
        bias_ptr = bias
    else:
        bias_ptr = c  # Dummy pointer

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    int8_gemm_addmm_quant_kernel[grid](
        a, b, c, c_s, bias_ptr, scale_a, scale_b, M, N, K,
        out_block_size,
        HAS_BIAS=has_bias,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128
    )

    # Reshape scales to match batch dimensions
    if len(batch_shape) > 0:
        c_s = c_s.reshape(*batch_shape, n_blocks)

    return c, c_s


# =============================================================================
# Fused INT8 GELU Activation
# =============================================================================


@triton.heuristics({
    "BLOCK_SN": lambda args: args["BLOCK_N"] // args["BLOCK_SIZE"],
})
@triton.jit
def int8_gelu_kernel(
    output_ptr, output_scale_ptr, input_ptr, input_scale_ptr,
    M, N: tl.constexpr, SN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SN: tl.constexpr,
):
    """Fused INT8 GELU with block-wise quantization.

    Computes: output_int8, output_scale = quantize(gelu(dequantize(input)))

    This avoids materializing the full-precision intermediate, keeping
    data in INT8 format throughout the FFN block.

    Args:
        output_ptr: Pointer to INT8 output tensor.
        output_scale_ptr: Pointer to output scales.
        input_ptr: Pointer to INT8 input tensor.
        input_scale_ptr: Pointer to input scales.
        M: Number of rows.
        N: Number of columns.
        SN: Number of scale blocks (N // BLOCK_SIZE).
        BLOCK_SIZE: Quantization block size (e.g., 128).
        BLOCK_M: Tile size in M dimension.
        BLOCK_N: Tile size in N dimension.
        BLOCK_SN: Number of scale blocks per tile (BLOCK_N // BLOCK_SIZE).
    """
    pid = tl.program_id(0)
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    pid_m = pid // NUM_BLOCK_N
    pid_n = pid % NUM_BLOCK_N

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load input data
    input_ptrs = input_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    input_data = tl.load(input_ptrs, mask=mask, other=0).to(tl.int32)

    # Load input scales
    offs_sn = pid_n * BLOCK_SN + tl.arange(0, BLOCK_SN)
    scale_ptrs = input_scale_ptr + offs_m[:, None] * SN + offs_sn[None, :]
    scale_mask = (offs_m[:, None] < M) & (offs_sn[None, :] < SN)
    input_scales = tl.load(scale_ptrs, mask=scale_mask, other=1.0)

    # Reshape for broadcasting: data (M, N) -> (M, SN, BLOCK_SIZE)
    input_data = tl.reshape(input_data, (BLOCK_M, BLOCK_SN, BLOCK_SIZE))
    input_scales = tl.reshape(input_scales, (BLOCK_M, BLOCK_SN, 1))

    # Dequantize
    input_fp32 = input_data.to(tl.float32) * input_scales

    # Apply GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    sqrt_2 = 1.41421356237
    erf_val = tl.math.erf(input_fp32 / sqrt_2)
    gelu_output = input_fp32 * 0.5 * (1.0 + erf_val)

    # Compute output scales per block
    abs_output = tl.abs(gelu_output)
    max_val = tl.max(abs_output, axis=2)
    output_scales = tl.maximum(max_val / 127.0, 1e-8)

    # Quantize output
    output_scales_broadcast = tl.reshape(output_scales, (BLOCK_M, BLOCK_SN, 1))
    quantized = gelu_output / output_scales_broadcast
    quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0)
    quantized_int8 = quantized.to(tl.int8)
    quantized_int8 = tl.reshape(quantized_int8, (BLOCK_M, BLOCK_N))

    # Store quantized output
    output_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(output_ptrs, quantized_int8, mask=mask)

    # Store output scales
    output_scale_ptrs = output_scale_ptr + offs_m[:, None] * SN + offs_sn[None, :]
    tl.store(output_scale_ptrs, output_scales, mask=scale_mask)


def int8_gelu(
    x: torch.Tensor,
    s_x: torch.Tensor,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused INT8 GELU activation with block-wise quantization.

    Computes: y_int8, y_scale = quantize(gelu(dequantize(x, s_x)))

    This avoids materializing the full-precision intermediate result.

    Args:
        x: INT8 input tensor of any shape.
        s_x: Input scales with shape (*batch_dims, last_dim // block_size).
        block_size: Quantization block size (default 128).

    Returns:
        Tuple of (quantized_output_int8, output_scales).
    """
    assert x.is_contiguous() and s_x.is_contiguous()
    assert x.size(-1) % block_size == 0

    # Determine BLOCK_N
    kernel_block_n = max(128, block_size)
    if kernel_block_n % block_size != 0:
        kernel_block_n = block_size

    # Handle multi-dimensional tensors by reshaping to 2D
    original_shape = x.shape
    batch_shape = original_shape[:-1]
    N = original_shape[-1]

    if x.dim() > 2:
        x = x.reshape(-1, N)
        s_x = s_x.reshape(-1, s_x.size(-1))

    M = x.size(0)
    SN = N // block_size

    # Allocate output tensors
    y = torch.empty_like(x, dtype=torch.int8)
    s_y = torch.empty_like(s_x, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    int8_gelu_kernel[grid](
        y, s_y, x, s_x, M, N, SN,
        BLOCK_SIZE=block_size,
        BLOCK_M=128, BLOCK_N=kernel_block_n,
    )

    # Reshape back to original batch dimensions
    if len(batch_shape) > 0:
        y = y.reshape(*batch_shape, N)
        s_y = s_y.reshape(*batch_shape, SN)

    return y, s_y


# =============================================================================
# INT8 Tensor-wise Quantization (from dxqb/OneTrainer)
# =============================================================================
# Simpler approach: single scale per tensor + per-row activation scaling.
# Uses torch._int_mm compatible format.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['QUANTIZED_M', 'N', 'K', 'stride_bk'],
)
@triton.jit
def mm_int8_tensorwise_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    QUANTIZED_M,
):
    """INT8 GEMM kernel: C[M,N] = A[M,K] @ B[K,N].

    Accumulates in int32. Output is int32 for later scaling.
    Based on dxqb/OneTrainer Triton kernel.
    """
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        b_mask = (offs_bn[None, :] < N) & (offs_k[:, None] < K - k * BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=a_mask, other=0)
        b = tl.load(b_ptrs, mask=b_mask, other=0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def mm_int8_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """INT8 matrix multiplication: C[M,N] = A[M,K] @ B[K,N].

    Uses autotuned Triton kernel. Output is int32 for later dequantization.

    Args:
        a: INT8 tensor [M, K].
        b: INT8 tensor [K, N].

    Returns:
        INT32 tensor [M, N] with accumulated dot products.
    """
    assert a.dtype == torch.int8 and b.dtype == torch.int8
    assert a.dim() == 2 and b.dim() == 2
    assert a.size(1) == b.size(0), f"K mismatch: {a.size(1)} vs {b.size(0)}"

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)

    grid = lambda META: (
        triton.cdiv(N, META['BLOCK_SIZE_N']),
        triton.cdiv(M, META['BLOCK_SIZE_M'])
    )
    mm_int8_tensorwise_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        QUANTIZED_M=M // 64,
    )
    return c


def quantize_int8_tensorwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with single tensorwise scale.

    Args:
        x: Input tensor of any shape.

    Returns:
        Tuple of (quantized_int8, scale):
            - quantized_int8: INT8 tensor with same shape
            - scale: Scalar float32 tensor
    """
    abs_max = x.abs().max()
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q = (x.float() / scale).round().clamp(-128.0, 127.0).to(torch.int8)
    return q, scale


def quantize_int8_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with per-row scales (for activations).

    Args:
        x: Input tensor [..., K] where quantization is per-row.

    Returns:
        Tuple of (quantized_int8, scales):
            - quantized_int8: INT8 tensor with same shape
            - scales: Float32 tensor [..., 1] with per-row scales
    """
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q = (x.float() / scale).round().clamp(-128.0, 127.0).to(torch.int8)
    return q, scale


def dequantize_int8_simple(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT8 tensor with scale.

    Args:
        q: Quantized INT8 tensor.
        scale: Scale tensor (scalar or broadcastable).

    Returns:
        Dequantized float tensor.
    """
    return q.float() * scale


def int8_linear_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """INT8 linear layer using Triton mm_int8 kernel.

    Quantizes x dynamically per-row, uses tensorwise weight scale.

    Args:
        x: Input tensor [..., K].
        weight: INT8 weight tensor [N, K].
        weight_scale: Scalar weight scale.
        bias: Optional bias [N].
        out_dtype: Output dtype.

    Returns:
        Result tensor [..., N].
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    # Quantize input per-row
    x_8, x_scale = quantize_int8_rowwise(x_2d)

    # Compute: x_8 @ weight.T using Triton kernel
    # weight is [N, K], we need [K, N] for matmul so transpose
    result = mm_int8_triton(x_8, weight.T.contiguous())

    # Scale back: result * (weight_scale * x_scale)
    result = result.float() * (weight_scale * x_scale)

    if bias is not None:
        result = result + bias.to(result.dtype)

    result = result.to(out_dtype)
    return result.reshape(*orig_shape[:-1], weight.shape[0])

