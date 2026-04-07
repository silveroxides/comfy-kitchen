# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Layered tests for SageAttention integration.

Layer 1: End-to-end SDPA vs torch reference
Layer 2: Standalone Q/K quantization, V quantization, CUDA kernel
Layer 3: Integration chain (standalone pieces = e2e)
Layer 4: Edge cases (device, compute capability, causal)
Layer 5: Slow model-grid sweep (pytest -m slow)
"""

import pytest
import torch
import torch.nn.functional as F

_CUDA = torch.cuda.is_available()
_SM89 = _CUDA and torch.cuda.get_device_capability() >= (8, 9)

try:
    from comfy_kitchen.backends.cuda import _EXT_AVAILABLE
except ImportError:
    _EXT_AVAILABLE = False

try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

_SAGE_READY = _SM89 and _EXT_AVAILABLE

requires_sage = pytest.mark.skipif(
    not _SAGE_READY,
    reason="Requires SM89+ GPU and compiled CUDA extension",
)

requires_triton = pytest.mark.skipif(
    not (_SAGE_READY and _HAS_TRITON),
    reason="Requires SM89+ GPU, CUDA extension, and Triton",
)

FAST_CONFIGS = [(2, 8, 128), (1, 24, 128)]
DTYPES = [torch.bfloat16, torch.float16]

MODEL_BHD: dict[str, list[int]] = {
    "black-forest-labs/FLUX.2-klein-4B": [1, 24, 128],
    "black-forest-labs/FLUX.2-klein-9B": [1, 32, 128],
    "black-forest-labs/FLUX.2-dev":      [1, 48, 128],
    "black-forest-labs/FLUX.1-dev":      [1, 24, 128],
    "Tongyi-MAI/Z-Image-Turbo":          [1, 30, 128],
    "Lightricks/LTX-2":                  [2, 32, 128],
    "Wan-AI/Wan2.2-I2V-A14B":            [1, 40, 128],
}
SEQ_LENGTHS = [4096, 8192, 16384, 24576, 32768]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref_sdpa(q, k, v, is_causal=False):
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def _make_qkv(B, H_Q, H_K, N_Q, N_K, D, dtype=torch.bfloat16, device="cuda"):
    q = torch.randn(B, H_Q, N_Q, D, dtype=dtype, device=device)
    k = torch.randn(B, H_K, N_K, D, dtype=dtype, device=device)
    v = torch.randn(B, H_K, N_K, D, dtype=dtype, device=device)
    return q, k, v


def _cos_sim(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0)


def _per_thread_int8_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    km: torch.Tensor | None = None,
    BLKQ: int = 128,
    WARPQ: int = 32,
    BLKK: int = 64,
    WARPK: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """INT8 per-thread quantization for Q and K (contiguous HND layout).

    Component-level test helper wrapping ``_C._quant_qk_per_thread_int8``.
    """
    from comfy_kitchen.backends.cuda import _C, _wrap_for_dlpack
    from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE

    if km is not None:
        k = k - km

    b, h_qo, qo_len, head_dim = q.shape
    _, h_kv, kv_len, _ = k.shape

    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    q_scale = torch.empty(
        (b, h_qo, (qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8),
        device=q.device, dtype=torch.float32,
    )
    k_scale = torch.empty(
        (b, h_kv, (kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4),
        device=k.device, dtype=torch.float32,
    )

    code = DTYPE_TO_CODE[q.dtype]
    if code not in (1, 2) or k.dtype != q.dtype:
        raise TypeError("q and k must be fp16 or bf16, same dtype")

    stream_ptr = torch.cuda.current_stream(q.device).cuda_stream
    _C._quant_qk_per_thread_int8(
        _wrap_for_dlpack(q),
        _wrap_for_dlpack(q_int8),
        _wrap_for_dlpack(q_scale),
        _wrap_for_dlpack(k),
        _wrap_for_dlpack(k_int8),
        _wrap_for_dlpack(k_scale),
        BLKQ, WARPQ, BLKK, WARPK,
        code, stream_ptr,
    )
    return q_int8, q_scale, k_int8, k_scale


# ---------------------------------------------------------------------------
# Layer 1: End-to-end SDPA
# ---------------------------------------------------------------------------

@requires_sage
class TestSageSDPAEndToEnd:
    """Compare sage_sdpa output against torch SDPA reference."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("D", [64, 128])
    def test_basic_accuracy(self, dtype, D):
        from comfy_kitchen.sage_attention import sage_sdpa

        B, H, N = 2, 8, 256
        q, k, v = _make_qkv(B, H, H, N, N, D, dtype=dtype)

        out = sage_sdpa(q, k, v, is_causal=False, smooth_k=True)
        ref = _ref_sdpa(q, k, v, is_causal=False)

        assert out.shape == ref.shape
        assert out.dtype == dtype
        assert _cos_sim(out, ref) > 0.95

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_gqa(self, dtype):
        from comfy_kitchen.sage_attention import sage_sdpa

        B, H_Q, H_K, N, D = 1, 16, 4, 128, 128
        q, k, v = _make_qkv(B, H_Q, H_K, N, N, D, dtype=dtype)

        out = sage_sdpa(q, k, v)
        assert out.shape == (B, H_Q, N, D)

        k_expanded = k.repeat_interleave(H_Q // H_K, dim=1)
        v_expanded = v.repeat_interleave(H_Q // H_K, dim=1)
        ref = _ref_sdpa(q, k_expanded, v_expanded)
        assert _cos_sim(out, ref) > 0.95

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_causal(self, dtype):
        from comfy_kitchen.sage_attention import sage_sdpa

        B, H, N, D = 1, 4, 256, 128
        q, k, v = _make_qkv(B, H, H, N, N, D, dtype=dtype)

        out = sage_sdpa(q, k, v, is_causal=True)
        ref = _ref_sdpa(q, k, v, is_causal=True)
        assert out.shape == ref.shape
        assert _cos_sim(out, ref) > 0.93

    def test_no_smooth_k(self):
        from comfy_kitchen.sage_attention import sage_sdpa

        B, H, N, D = 1, 4, 128, 64
        q, k, v = _make_qkv(B, H, H, N, N, D)

        out = sage_sdpa(q, k, v, smooth_k=False)
        ref = _ref_sdpa(q, k, v)
        assert _cos_sim(out, ref) > 0.90


# ---------------------------------------------------------------------------
# Layer 2: Standalone components
# ---------------------------------------------------------------------------

@requires_sage
class TestQKQuantizationCUDA:
    """Test per-thread INT8 quantization for Q and K (CUDA kernel)."""

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_shapes_and_dtypes(self, dtype):
        B, H, N, D = 2, 8, 256, 128
        q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
        k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")

        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_cuda(q, k)

        assert q_int8.shape == q.shape
        assert q_int8.dtype == torch.int8
        assert k_int8.shape == k.shape
        assert k_int8.dtype == torch.int8
        assert q_scale.dtype == torch.float32
        assert k_scale.dtype == torch.float32

    def test_scale_shapes(self):
        B, H_Q, H_K, N_Q, N_K, D = 1, 8, 4, 256, 128, 64
        q = torch.randn(B, H_Q, N_Q, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_K, N_K, D, dtype=torch.bfloat16, device="cuda")

        BLKQ, WARPQ, BLKK, WARPK = 128, 32, 64, 64
        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_cuda(q, k)

        expected_q_scales = ((N_Q + BLKQ - 1) // BLKQ) * (BLKQ // WARPQ) * 8
        expected_k_scales = ((N_K + BLKK - 1) // BLKK) * (BLKK // WARPK) * 4

        assert q_scale.shape == (B, H_Q, expected_q_scales)
        assert k_scale.shape == (B, H_K, expected_k_scales)

    def test_roundtrip_bounded(self):
        B, H, N, D = 1, 4, 128, 128
        q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")

        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_cuda(q, k)
        assert _cos_sim(q, q_int8.float()) > 0.8

    def test_smooth_k(self):
        B, H, N, D = 1, 4, 128, 64
        q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
        km = k.mean(dim=2, keepdim=True)

        _, _, k_int8_smooth, _ = _per_thread_int8_cuda(q, k, km=km)
        _, _, k_int8_raw, _ = _per_thread_int8_cuda(q, k, km=None)
        assert not torch.equal(k_int8_smooth, k_int8_raw)


@requires_triton
class TestQKQuantizationTritonRef:
    """Optional: compare CUDA Q/K quant against Triton reference."""

    def test_cuda_matches_triton(self):
        from tests.sage_triton_ref.quant_per_thread import per_thread_int8

        B, H, N, D = 1, 4, 256, 128
        q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")

        cuda_out = _per_thread_int8_cuda(q, k)
        triton_out = per_thread_int8(q, k, tensor_layout="HND")

        for c, t in zip(cuda_out, triton_out):
            torch.testing.assert_close(c, t, atol=1, rtol=0)


@requires_sage
class TestVQuantization:
    """Test FP8 per-channel V quantization."""

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_shapes_and_dtypes(self, dtype):
        from comfy_kitchen.sage_attention import CTA_K, _quantize_v_fp8

        B, H, N, D = 2, 4, 256, 128
        v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
        v_quant, v_scale = _quantize_v_fp8(v)
        padded_N = ((N + CTA_K - 1) // CTA_K) * CTA_K

        assert v_quant.shape == (B, H, D, padded_N)
        assert v_quant.dtype == torch.int8
        assert v_scale.shape == (B, H, D)
        assert v_scale.dtype == torch.float32

    def test_padding(self):
        from comfy_kitchen.sage_attention import CTA_K, _quantize_v_fp8

        B, H, N, D = 1, 1, 100, 64
        v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
        v_quant, _ = _quantize_v_fp8(v)
        padded_N = ((N + CTA_K - 1) // CTA_K) * CTA_K
        assert v_quant.shape[3] == padded_N
        assert padded_N == 128

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_scale_positivity(self, dtype):
        from comfy_kitchen.sage_attention import _quantize_v_fp8

        B, H, N, D = 1, 2, 128, 64
        v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
        _, v_scale = _quantize_v_fp8(v)
        assert (v_scale > 0).all()


@requires_sage
class TestCUDAKernelStandalone:
    """Test the attention CUDA kernel in isolation with pre-quantized inputs."""

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_kernel_runs(self, dtype):
        from comfy_kitchen.backends.cuda import _C, _wrap_for_dlpack
        from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE
        from comfy_kitchen.sage_attention import _quantize_v_fp8

        B, H, N, D = 1, 4, 128, 128
        q, k, v = _make_qkv(B, H, H, N, N, D, dtype=dtype)

        km = k.mean(dim=2, keepdim=True)
        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_cuda(q, k, km=km)
        v_quant, v_scale = _quantize_v_fp8(v)

        output = torch.empty(B, H, N, D, dtype=dtype, device="cuda")
        _C._sage_attn(
            _wrap_for_dlpack(q_int8),
            _wrap_for_dlpack(k_int8),
            _wrap_for_dlpack(v_quant.contiguous()),
            _wrap_for_dlpack(output),
            _wrap_for_dlpack(q_scale),
            _wrap_for_dlpack(k_scale),
            _wrap_for_dlpack(v_scale),
            0,
            D ** -0.5,
            DTYPE_TO_CODE[dtype],
            torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()

        assert output.shape == (B, H, N, D)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_kernel_accuracy(self, dtype):
        from comfy_kitchen.backends.cuda import _C, _wrap_for_dlpack
        from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE
        from comfy_kitchen.sage_attention import _quantize_v_fp8

        B, H, N, D = 1, 4, 256, 128
        q, k, v = _make_qkv(B, H, H, N, N, D, dtype=dtype)

        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_cuda(q, k)
        v_quant, v_scale = _quantize_v_fp8(v)

        output = torch.empty(B, H, N, D, dtype=dtype, device="cuda")
        _C._sage_attn(
            _wrap_for_dlpack(q_int8),
            _wrap_for_dlpack(k_int8),
            _wrap_for_dlpack(v_quant.contiguous()),
            _wrap_for_dlpack(output),
            _wrap_for_dlpack(q_scale),
            _wrap_for_dlpack(k_scale),
            _wrap_for_dlpack(v_scale),
            0,
            D ** -0.5,
            DTYPE_TO_CODE[dtype],
            torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()

        ref = _ref_sdpa(q, k, v)
        assert _cos_sim(output, ref) > 0.90


# ---------------------------------------------------------------------------
# Layer 3: Integration (chain matches e2e)
# ---------------------------------------------------------------------------

@requires_sage
class TestIntegrationChain:
    """Verify that chaining standalone pieces matches sage_sdpa output."""

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_chain_matches_e2e(self, dtype):
        from comfy_kitchen.backends.cuda import _C, _wrap_for_dlpack
        from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE
        from comfy_kitchen.sage_attention import _quantize_v_fp8, sage_sdpa

        B, H, N, D = 1, 8, 256, 128

        torch.manual_seed(42)
        q, k, v = _make_qkv(B, H, H, N, N, D, dtype=dtype)

        torch.manual_seed(42)
        e2e_out = sage_sdpa(q, k, v, is_causal=False, smooth_k=True)

        km = k.mean(dim=2, keepdim=True)
        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_cuda(q, k, km=km)
        v_quant, v_scale = _quantize_v_fp8(v)

        chain_out = torch.empty_like(e2e_out)
        _C._sage_attn(
            _wrap_for_dlpack(q_int8),
            _wrap_for_dlpack(k_int8),
            _wrap_for_dlpack(v_quant.contiguous()),
            _wrap_for_dlpack(chain_out),
            _wrap_for_dlpack(q_scale),
            _wrap_for_dlpack(k_scale),
            _wrap_for_dlpack(v_scale),
            0,
            D ** -0.5,
            DTYPE_TO_CODE[dtype],
            torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(e2e_out, chain_out, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Layer 4: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_is_available_type(self):
        from comfy_kitchen.sage_attention import is_available
        assert isinstance(is_available(), bool)

    @pytest.mark.skipif(not _CUDA, reason="CUDA not available")
    @pytest.mark.skipif(not _SM89, reason="Needs SM89+")
    @pytest.mark.skipif(not _EXT_AVAILABLE, reason="CUDA extension not built")
    def test_is_available_on_sm89(self):
        from comfy_kitchen.sage_attention import is_available
        assert is_available() is True

    @requires_sage
    def test_bad_head_dim(self):
        from comfy_kitchen.sage_attention import sage_sdpa
        q, k, v = _make_qkv(1, 4, 4, 128, 128, 96)
        with pytest.raises(AssertionError, match="head_dim must be 64 or 128"):
            sage_sdpa(q, k, v)

    @requires_sage
    def test_mismatched_heads(self):
        from comfy_kitchen.sage_attention import sage_sdpa
        q = torch.randn(1, 7, 128, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 3, 128, 64, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, 3, 128, 64, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(AssertionError, match="must be divisible"):
            sage_sdpa(q, k, v)

    @requires_sage
    def test_public_api_reachable(self):
        import comfy_kitchen
        assert hasattr(comfy_kitchen, "sage_sdpa")
        assert callable(comfy_kitchen.sage_sdpa)


# ---------------------------------------------------------------------------
# Layer 5: Slow model-grid sweep
# ---------------------------------------------------------------------------

_SLOW_CONFIGS = [
    pytest.param(name, bhd, seq, id=f"{name.split('/')[-1]}-N{seq}")
    for name, bhd in MODEL_BHD.items()
    for seq in SEQ_LENGTHS
]


@requires_sage
@pytest.mark.slow
class TestModelGridSweep:
    """Run sage_sdpa across real model configs; only with ``pytest -m slow``."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("model_name,bhd,seq_len", _SLOW_CONFIGS)
    def test_sdpa_accuracy(self, model_name, bhd, seq_len, dtype):
        from comfy_kitchen.sage_attention import sage_sdpa

        B, H, D = bhd
        q, k, v = _make_qkv(B, H, H, seq_len, seq_len, D, dtype=dtype)

        out = sage_sdpa(q, k, v, is_causal=False)
        ref = _ref_sdpa(q, k, v, is_causal=False)

        assert out.shape == ref.shape
        assert _cos_sim(out, ref) > 0.93, (
            f"{model_name} B={B} H={H} N={seq_len} D={D} {dtype}: "
            f"cosine similarity {_cos_sim(out, ref):.4f} too low"
        )
