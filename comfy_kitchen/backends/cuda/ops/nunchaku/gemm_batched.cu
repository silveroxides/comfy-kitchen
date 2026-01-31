#include "gemm_batched.h"

#include "dispatch_cutlass.h"

#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>

namespace nunchaku {

using ElementInput  = cutlass::half_t;
using ElementOutput = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutO = cutlass::layout::RowMajor;

using Gemm = cutlass::gemm::device::GemmBatched<
    ElementInput,
    LayoutA,
    ElementInput,
    LayoutB,
    ElementOutput,
    LayoutO,
    ElementOutput,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                 128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                 ElementOutput,
                                                 ElementOutput>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2>;

size_t gemm_batched_fp16_get_workspace_size(Tensor a, Tensor b, Tensor out) {
    const int M     = a.shape[-2];
    const int K     = a.shape[-1];
    
    int N = 0;
    if (out.valid()) {
        N = out.shape[-1];
    } else {
        N = a.shape[-2];
    }
    int batch = (int)(a.numel() / (M * K));

    cutlass::gemm::GemmCoord problemSize(M, N, K);

    cutlass::TensorRef<ElementInput, LayoutA> refA(nullptr, LayoutA((int64_t)a.stride(-2)));
    cutlass::TensorRef<ElementInput, LayoutB> refB(nullptr, LayoutB((int64_t)b.stride(-2)));
    cutlass::TensorRef<ElementOutput, LayoutO> refO(nullptr, LayoutO(out.valid() ? (int64_t)out.stride(-2) : (int64_t)N));

    typename Gemm::Arguments arguments{problemSize,
                                       refA,
                                       (int64_t)a.stride(-3),
                                       refB,
                                       (int64_t)b.stride(-3),
                                       refO,
                                       (int64_t)(out.valid() ? out.stride(-3) : M*N),
                                       refO,
                                       (int64_t)(out.valid() ? out.stride(-3) : M*N),
                                       {ElementOutput(1), ElementOutput(0)},
                                       batch};

    return Gemm::get_workspace_size(arguments);
}

void gemm_batched_fp16(Tensor a,  // FP16 row-major [(... batch ...), M, K]
                       Tensor b,  // FP16 col-major [(... batch ...), N, K]
                       Tensor out, // FP32 row-major [(... batch ...), M, N]
                       void* workspace,
                       size_t workspace_size
) {
    const int M     = a.shape[-2];
    const int K     = a.shape[-1];
    const int N     = out.valid() ? out.shape[-1] : a.shape[-2];
    const int batch = (int)(a.numel() / (M * K));

    if (!out.valid()) {
        throw std::runtime_error("Output tensor must be pre-allocated for gemm_batched_fp16");
    }
    
    assert(K == b.shape[-1]);
    assert(M == out.shape[-2]);

    assert(a.dtype() == Tensor::FP16);
    assert(a.dtype() == b.dtype());
    assert(out.dtype() == Tensor::FP32);

    cutlass::gemm::GemmCoord problemSize(M, N, K);

    cutlass::TensorRef<ElementInput, LayoutA> refA(a.data_ptr<ElementInput>(), LayoutA((int64_t)a.stride(-2)));
    cutlass::TensorRef<ElementInput, LayoutB> refB(b.data_ptr<ElementInput>(), LayoutB((int64_t)b.stride(-2)));
    cutlass::TensorRef<ElementOutput, LayoutO> refO(out.data_ptr<ElementOutput>(), LayoutO((int64_t)out.stride(-2)));

    typename Gemm::Arguments arguments{problemSize,
                                       refA,
                                       (int64_t)a.stride(-3),
                                       refB,
                                       (int64_t)b.stride(-3),
                                       refO,
                                       (int64_t)out.stride(-3),
                                       refO,
                                       (int64_t)out.stride(-3),
                                       {ElementOutput(1), ElementOutput(0)},
                                       batch};

    Gemm op;
    size_t required_workspace = Gemm::get_workspace_size(arguments);
    
    if (required_workspace > workspace_size) {
        throw std::runtime_error("Insufficient workspace for gemm_batched");
    }
    
    cudaStream_t stream = getCurrentCUDAStream();

    cutlass::Status status = op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot implement gemm_batched");
    }

    status = op.initialize(arguments, workspace);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot initialize");
    }

    status = op(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot run");
    }

}

} // namespace nunchaku
