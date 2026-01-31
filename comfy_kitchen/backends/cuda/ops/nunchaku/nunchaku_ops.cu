#include "gemm_w4a4_launch.cuh"
#include "gemm_batched.h"
#include "common.h"

namespace nunchaku {
    // Helper to wrap pointer into Tensor
    Tensor make_tensor(const void* ptr, std::vector<int> shape, int dtype_code) {
        Tensor t;
        t.ptr = (void*)ptr;
        t.shape.dataExtent = shape;
        
        switch(dtype_code) {
            case 0: t.scalarType = Tensor::FP32; break;
            case 1: t.scalarType = Tensor::FP16; break;
            case 2: t.scalarType = Tensor::BF16; break;
            case 3: t.scalarType = Tensor::INT8; break; // uint8 mapped to int8
            case 4: t.scalarType = Tensor::INT8; break;
            case 5: t.scalarType = Tensor::FP8_E4M3; break;
            case 6: t.scalarType = Tensor::FP8_E5M2; break;
            default: t.scalarType = Tensor::INVALID_SCALAR_TYPE;
        }
        return t;
    }

    void launch_gemm_w4a4_impl(
        void* output,
        const void* input,
        const void* weight,
        const void* wscales,
        const void* ascales,
        const void* bias,
        int M,
        int N,
        int K,
        int input_dtype_code,
        int weight_dtype_code,
        int wscales_dtype_code,
        int ascales_dtype_code,
        int bias_dtype_code,
        cudaStream_t stream
    ) {
        CUDAStreamContext stream_ctx(stream);

        Tensor t_output = make_tensor(output, {M, N}, input_dtype_code); // Assuming output type similar to input/standard
        Tensor t_input = make_tensor(input, {M, K/2}, input_dtype_code);
        Tensor t_weight = make_tensor(weight, {N, K/2}, weight_dtype_code);
        
        Tensor t_wscales = make_tensor(wscales, {K/64, N}, wscales_dtype_code); 
        Tensor t_ascales;
        if (ascales) {
            t_ascales = make_tensor(ascales, {K/64, M}, ascales_dtype_code);
        }
        
        Tensor t_bias;
        if (bias) {
             t_bias = make_tensor(bias, {N}, bias_dtype_code);
        }

        // Empty tensors for unused args
        Tensor t_qout, t_oscales, t_poolout;
        Tensor t_lora_act_in, t_lora_up, t_lora_down, t_lora_act_out;
        Tensor t_norm_q, t_norm_k, t_rotary_emb, t_smooth_factor, t_out_vk, t_out_linearattn;
        Tensor t_wcscales, t_out_q, t_out_k, t_out_v;

        using namespace kernels;

        if (input_dtype_code == 1) { // FP16
             GEMM_W4A4_Launch<GEMMConfig_W4A4_FP16, false>::gemm_w4a4(
                t_input, t_weight, t_output, t_qout,
                t_ascales, t_wscales, t_oscales, t_poolout,
                t_lora_act_in, t_lora_up, t_lora_down, t_lora_act_out,
                t_norm_q, t_norm_k, t_rotary_emb, t_bias,
                t_smooth_factor, t_out_vk, t_out_linearattn,
                false, // act_unsigned
                {}, // lora_scales
                false, // fuse_silu
                false, // fp4
                1.0f, // alpha
                t_wcscales, t_out_q, t_out_k, t_out_v,
                0 // attn_tokens
             );
        } else if (input_dtype_code == 2) { // BF16
             GEMM_W4A4_Launch<GEMMConfig_W4A4_BF16, false>::gemm_w4a4(
                t_input, t_weight, t_output, t_qout,
                t_ascales, t_wscales, t_oscales, t_poolout,
                t_lora_act_in, t_lora_up, t_lora_down, t_lora_act_out,
                t_norm_q, t_norm_k, t_rotary_emb, t_bias,
                t_smooth_factor, t_out_vk, t_out_linearattn,
                false, // act_unsigned
                {}, // lora_scales
                false, // fuse_silu
                false, // fp4
                1.0f, // alpha
                t_wcscales, t_out_q, t_out_k, t_out_v,
                0 // attn_tokens
             );
        }
    }

    size_t gemm_batched_workspace_size_impl(
        int batch, int M, int N, int K,
        int dtype_code
    ) {
        // Construct dummy tensors for size calculation
        Tensor t_a; t_a.shape.dataExtent = {batch, M, K}; t_a.scalarType = (Tensor::ScalarType)dtype_code; 
        Tensor t_b; t_b.shape.dataExtent = {batch, N, K}; t_b.scalarType = (Tensor::ScalarType)dtype_code;
        Tensor t_out; t_out.shape.dataExtent = {batch, M, N}; t_out.scalarType = Tensor::FP32; // Output is FP32
        
        if (dtype_code == 1) { t_a.scalarType = Tensor::FP16; t_b.scalarType = Tensor::FP16; }
        
        return gemm_batched_fp16_get_workspace_size(t_a, t_b, t_out);
    }

    void launch_gemm_batched_impl(
        const void* a, const void* b, void* out, void* workspace, size_t workspace_size,
        int batch, int M, int N, int K,
        int dtype_code,
        cudaStream_t stream
    ) {
        CUDAStreamContext stream_ctx(stream);

        Tensor t_a = make_tensor(a, {batch, M, K}, dtype_code);
        Tensor t_b = make_tensor(b, {batch, N, K}, dtype_code);
        Tensor t_out = make_tensor(out, {batch, M, N}, 0); // FP32 output
        t_out.scalarType = Tensor::FP32;

        gemm_batched_fp16(t_a, t_b, t_out, workspace, workspace_size);
    }
}

extern "C" {
    size_t nunchaku_gemm_batched_workspace_size(
        int batch, int M, int N, int K, int dtype_code
    ) {
        return nunchaku::gemm_batched_workspace_size_impl(batch, M, N, K, dtype_code);
    }

    void launch_nunchaku_gemm_batched_kernel(
        const void* a, const void* b, void* out, void* workspace, size_t workspace_size,
        int batch, int M, int N, int K,
        int dtype_code,
        cudaStream_t stream
    ) {
        nunchaku::launch_gemm_batched_impl(a, b, out, workspace, workspace_size, batch, M, N, K, dtype_code, stream);
    }

    void launch_nunchaku_gemm_w4a4_kernel(
        const void* input,
        const void* weight,
        const void* wscales,
        const void* ascales,
        const void* bias,
        void* output,
        int M,
        int N,
        int K,
        int input_dtype_code,
        int weight_dtype_code,
        int wscales_dtype_code,
        int ascales_dtype_code,
        int bias_dtype_code,
        cudaStream_t stream
    ) {
        nunchaku::launch_gemm_w4a4_impl(
            output, input, weight, wscales, ascales, bias,
            M, N, K,
            input_dtype_code, weight_dtype_code, wscales_dtype_code, ascales_dtype_code, bias_dtype_code,
            stream
        );
    }
}
