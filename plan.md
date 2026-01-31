# Nunchaku Kernel Implementation Plan for Comfy-Kitchen

## Context
This project involves porting high-performance CUDA kernels from the **Nunchaku** repository to the **Comfy-Kitchen** CUDA backend. The goal is to provide a memory-safe and architecturally flexible implementation that decouples from PyTorch's C++ API while maintaining compatibility with Comfy-Kitchen's DLPack-based infrastructure.

## Repositories
- **Nunchaku Root**: `F:\repos\nunchaku`
- **Comfy-Kitchen Root**: `F:\repos\comfy-kitchen`
- **Target Backend Directory**: `F:\repos\comfy-kitchen\comfy_kitchen\backends\cuda\ops\nunchaku\`

## Accomplished So Far
1.  **Environment Setup**: Created the `nunchaku` ops directory within the Comfy-Kitchen CUDA backend.
2.  **Memory Abstraction**:
    - Created `Tensor.h`: A lightweight tensor wrapper that replaces Nunchaku's original `Tensor` class. It manages raw pointers and shapes without depending on PyTorch.
    - Created `common.h`: A cleaned-up version of Nunchaku's common helpers, removing `spdlog` and isolating CUDA/cuBLAS error handling.
3.  **Kernel Utilities**:
    - Ported `utils.cuh`, `dispatch_utils.h`, `gemm_utils.cuh`, and `mma_earlycuda.cuh`. These provide the foundation for dispatching kernels and performing low-level MMA (Matrix Multiply-Accumulate) operations.
4.  **GEMM Foundation**:
    - Ported `gemm_base.cuh` and `lora.cuh`. These define the base classes for W4A4 GEMM and the LoRA epilogues.

## Immediate Next Steps (For Next Agent)
1.  **Complete W4A4 GEMM Port**:
    - Copy and adapt `epilogues.cuh` from Nunchaku.
    - Copy and adapt `gemm_w4a4.cuh`, `gemm_w4a4_launch.cuh`, and `gemm_w4a4_launch_impl.cuh`.
    - **Note**: Ensure all `#include` paths in these files point to the local `nunchaku/` directory within `comfy-kitchen`.
2.  **Export Interface**:
    - Create `nunchaku_ops.cu` in `comfy_kitchen/backends/cuda/ops/`. This file should:
        - Include the launch headers.
        - Define `extern "C"` functions (e.g., `launch_nunchaku_gemm_w4a4_kernel`) that take raw pointers and metadata.
        - Map these functions to the `GEMM_W4A4_Launch` template calls.
3.  **Python Bindings**:
    - Update `F:\repos\comfy-kitchen\comfy_kitchen\backends\cuda\dlpack_bindings.cpp` to expose the new Nunchaku kernels to Python via nanobind.
4.  **Build Configuration**:
    - Update `F:\repos\comfy-kitchen\comfy_kitchen\backends\cuda\CMakeLists.txt` to include the new source files and ensure the `nunchaku/` directory is in the include path.
5.  **Additional Kernels**:
    - Port Activation (`activation_kernels.cu`) and LayerNorm (`layernorm_kernels.cu`) following the same pattern.
    - Port Attention kernels (`attention.cu`).

## Architectural Goals
- **Memory Safety**: Use the provided `Tensor` wrapper to avoid raw pointer arithmetic errors where possible, while keeping the kernels efficient.
- **Flexibility**: Ensure the C++ interface is pure and can be tested independently of the Python bindings.
