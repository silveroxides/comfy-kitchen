#pragma once

#include "common.h"
#include "Tensor.h"

namespace nunchaku {

size_t gemm_batched_fp16_get_workspace_size(Tensor a, Tensor b, Tensor out);

void gemm_batched_fp16(Tensor a,  // FP16 row-major [(... batch ...), M, K]
                       Tensor b,  // FP16 col-major [(... batch ...), N, K]
                       Tensor out, // FP32 row-major [(... batch ...), M, N]
                       void* workspace,
                       size_t workspace_size
);

} // namespace nunchaku
