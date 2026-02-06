"""
    Triton implementation of the Flash Attention v2 algorithm (Blackwell version, forward pass only)
    Key changes:
    1. Supports FP8 inputs and outputs (on Blackwell GPUs)
    2. Block pointers for better memory access patterns on Blackwell
"""

import pytest
import torch
import triton
import triton.language as tl

from utils import configs, keep, prune_invalid_configs, prune_invalid_configs
from utils import DEVICE, is_blackwell

@triton.jit
def _attn_fwd_inner(acc, # accumulator
                    d_i, # running denominator
                    m_i, # running max
                    q,  # query block
                    desc_k, desc_v,  # key and value
                    offset_y, 
                    dtype, 
                    start_m, 
                    qk_scale,  #
                    BLOCK_M: tl.constexpr, 
                    HEAD_DIM: tl.constexpr, 
                    BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, 
                    offs_m: tl.constexpr, 
                    offs_n: tl.constexpr,  #
                    N_CTX):
    
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo

    y_dim = tl.num_programs(1) * N_CTX

    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # --- K BLOCK POINTER ---
        # We point to the specific block of K based on start_n
        k_ptr = tl.make_block_ptr(
            base=desc_k,
            shape=(y_dim, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(offset_y + start_n, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0)
        )
        k = tl.load(k_ptr).T # Load and Transpose for dot product

        # --- V BLOCK POINTER ---
        v_ptr = tl.make_block_ptr(
            base=desc_v,
            shape=(y_dim, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(offset_y + start_n, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0)
        )
        v = tl.load(v_ptr)

        # -- compute qk ----
        qk = tl.dot(q, k) # compute qk^T

        if STAGE == 2: # causal masking
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            # -- get the new max
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        # -- compute current exponent --
        p = tl.math.exp2(qk)

        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        d_ij = tl.sum(p, 1) # running sum for the block

        # -- update output accumulator --
        acc = acc * alpha[:, None]

        p = p.to(dtype)

        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)

        # update m_i and d_i
        # place this at the end of the loop to reduce register pressure
        d_i = d_i * alpha + d_ij
        m_i = m_ij

        # update offsets
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N

    return acc, d_i, m_i

@triton.autotune(configs=list(filter(keep, configs)), 
                 key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
                 prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def _attn_fwd(sm_scale, M,  #
              Z, H, 
              desc_q, desc_k, desc_v, desc_o,  #
              N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    # Create block ptrs for blackwell.
    q_ptr = tl.make_block_ptr(
        base=desc_q,  # This is the raw pointer to Q
                shape=(y_dim, HEAD_DIM),
                strides=(HEAD_DIM, 1),
                offsets=(qo_offset_y, 0),
                block_shape=(BLOCK_M, HEAD_DIM),
                order=(1, 0)
            )
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    d_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    if STAGE & 1:
        acc, d_i, m_i = _attn_fwd_inner(acc, d_i, m_i, 
                                        q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, 
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX)
    # stage 2: on-band
    if STAGE & 2:
        acc, d_i, m_i = _attn_fwd_inner(acc, d_i, m_i, 
                                        q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, 
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX)
    # epilogue
    m_i += tl.math.log2(d_i)
    acc = acc / d_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    # Store O (Corrected Store)
    o_ptr = tl.make_block_ptr(
        base=desc_o,
        shape=(y_dim, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(qo_offset_y, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    tl.store(o_ptr, acc.to(dtype))


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        desc_q = q
        desc_v = v
        desc_k = k
        desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        ctx.grid = grid
        if is_blackwell():
            if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80

        _attn_fwd[grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            desc_q, desc_k, desc_v, desc_o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            STAGE=stage,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

attention = _attention.apply

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')

@pytest.mark.parametrize("Z", [1, 4])
@pytest.mark.parametrize("H", [2, 48])
@pytest.mark.parametrize("N_CTX", [128, 1024, (4) * 1024])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mode", ["fwd"])
@pytest.mark.parametrize("provider", ["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []))
def test_op(Z, H, N_CTX, HEAD_DIM, causal, mode, provider, dtype=torch.float16):

    if mode == "fwd" and "fp16" in provider:
        pytest.skip("Avoid running the forward computation twice.")
        
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    # reference implementation
    ref_dtype = dtype
    if mode == "fwd" and "fp8" in provider:
        ref_dtype = torch.float32
    q = q.to(ref_dtype)
    k = k.to(ref_dtype)
    v = v.to(ref_dtype)
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1)
    p = p.to(ref_dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v).half()
    if mode == "fwd" and "fp8" in provider:
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.permute(0, 1, 3, 2).contiguous()
        v = v.permute(0, 1, 3, 2)
        v = v.to(torch.float8_e5m2)
    tri_out = attention(q, k, v, causal, sm_scale).half()
    if mode == "fwd":
        atol = 3 if "fp8" in provider else 1e-2
        torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
        return
    # compare
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


BATCH, N_HEADS = 4, 32

# vary seq length for fixed head and batch=4
configs = []
for HEAD_DIM in [64, 128]:
    for mode in ["fwd"]:
        for causal in [True, False]:
            # Enable warpspec for causal fwd on Hopper
            enable_ws = mode == "fwd" and (is_blackwell() and causal)
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    x_vals=[2**i for i in range(10, 15)],
                    line_arg="provider",
                    line_vals=["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []) +
                    ([]),
                    line_names=["Triton [FP16]"] + (["Triton [FP8]"] if TORCH_HAS_FP8 else []) +
                    ([]),
                    styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                    ylabel="TFLOPS",
                    plot_name=
                    f"flash-attn-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                    args={
                        "H": N_HEADS,
                        "BATCH": BATCH,
                        "HEAD_DIM": HEAD_DIM,
                        "mode": mode,
                        "causal": causal,
                    },
                ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, 
                          causal, mode, provider, device=DEVICE):
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale)
        ms = triton.testing.do_bench(fn)

    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    bench_flash_attention.run(save_path="/home/ypzhao/code/kernel_projects/triton_kernels/results", print_data=True)
