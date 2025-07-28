# TLX GDPA kernel optimized for Blackwell Warp Specialization

import torch

import triton
import triton.language as tl
import triton.tlx.language as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
                "NUM_SMEM_BUFFERS": s,
                "NUM_TMEM_BUFFERS": t,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=matmul_tma_set_block_size_hook,
        )
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in [2, 3, 4]
        for t in [2, 3]
        for subtile in [True]
    ]


@triton.jit
def _compute_qlen(tile_idx, n_tile_num, Q_offsets, seq_index, SORT_BY_SEQ_LENGTH: tl.constexpr):
    off_hz = tile_idx // n_tile_num
    off_z = off_hz // H
    if SORT_BY_SEQ_LENGTH:
        off_z = tl.load(seq_index + off_z)
    off_q_z = off_z
    begin_q = tl.load(Q_offsets + off_q_z)
    end_q = tl.load(Q_offsets + off_q_z + 1)

    qlen = end_q - begin_q
    qlen = tl.minimum(qlen, N_CTX)
    return begin_q, qlen


@triton.jit
def _load_tma(bufIdx, phase, empty_bars, full_bars, buffers, desc, offset_1, offset_0, num_bytes):
    empty = tlx.local_view(empty_bars, bufIdx)
    tlx.barrier_wait(empty, phase ^ 1)
    full = tlx.local_view(full_bars, bufIdx)
    tlx.barrier_expect_bytes(full, num_bytes)
    view_smem = tlx.local_view(buffers, bufIdx)
    tlx.async_descriptor_load(desc, view_smem, [
        (offset_1).to(tl.int32),
        (offset_0).to(tl.int32),
    ], full)

    return view_smem


# Barriers:
#   producer_acquire uses the same barrier as consumer_release
#   producer_commit uses the same barriers as consumer_wait
@triton.jit
def _do_dots():
    # prologue
    accm_cnt_k = 0
    tlx.barrier_wait(consumer_q0)  # consumer wait for q0
    tlx.barrier_wait(consumer_k)  # consumer wait for k
    tlx.barrier_wait(producer_qk0)  # producer acquire for qk0
    # Do we support use_acc for async_dot?
    # producer commit for qk0
    tlx.async_dot(q0_view, k_view, qk0_view, use_acc=False, mBarriers=[producer_commit_qk0])
    tlx.barrier_wait(consumer_q1)  # consumer wait for q1
    tlx.barrier_wait(producer_qk1)  # producer acquire for qk1
    # consumer release for k, producer commit for qk1
    tlx.async_dot(q1_view, k_view, qk1_view, use_acc=False, mBarriers=[consumer_release_k, producer_commit_qk1])

    tlx.barrier_wait(consumer_v)  # consumer wait for v
    tlx.barrier_wait(producer_o0)  # producer acquire for o0
    tlx.barrier_wait(consumer_qk0)  # consumer wait for p0 due to reuse of p0 and qk0
    tlx.async_dot(p0_view, v_view, o0_view, use_acc=False, mBarriers=[producer_commit_o0])

    lo, hi = 0, klen
    first = True
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # q0 dot k
        tlx.barrier_wait(consumer_k)  # consumer wait for k
        # no need for acquiring qk0?
        tlx.async_dot(q0_view, k_view, qk0_view, use_acc=False, mBarriers=[producer_commit_qk0])
        # p1 dot v
        tlx.barrier_wait(producer_o1)  # producer acquire for o1
        tlx.barrier_wait(consumer_qk1)  # consumer wait for p1 due to reuse of p1 and qk1
        # done using v from previous iteration
        tlx.async_dot(p1_view, v_view, o1_view, use_acc=not first, mBarriers=[producer_commit_o1, consumer_release_v])
        # q1 dot k, done using k for this iteration
        tlx.async_dot(q1_view, k_view, qk1_view, use_acc=False, mBarriers=[consumer_release_k, producer_commit_qk1])
        # p0 dot v
        tlx.barrier_wait(consumer_v)  # consumer wait for v
        tlx.barrier_wait(producer_o0)  # producer acquire for o0
        tlx.async_dot(p0_view, v_view, o0_view, use_acc=True, mBarriers=[producer_commit_o0])
        first = False
    # epilogue
    # commit/commit
    tlx.barrier_wait(producer_o1)  # producer acquire for o1
    tlx.barrier_wait(consumer_qk1)  # consumer wait for p1 due to reuse of p1 and qk1
    # release p0, p1
    tlx.async_dot(p1_view, v_view, o1_view, use_acc=not first,
                  mBarriers=[producer_commit_o1, consumer_release_v, consumer_release_qk0, consumer_release_qk1])
    return


@triton.jit
def _do_activation(qk, qk_scale, v_dtype, activation_enum_int: tl.constexpr):
    # qk in tmem, output p in tmem
    # ConsumerWait for qk, ProducerAcquire for p
    # activation = gelu
    if activation_enum_int == 0:
        p = raw(qk)
    elif activation_enum_int == 1:
        p = gelu(qk)
    elif activation_enum_int == 2:
        p = gelu_approx(qk)
    elif activation_enum_int == 3:
        p = fast_gelu(qk)
    elif activation_enum_int == 4:
        p = leaky_relu(qk)
    elif activation_enum_int == 5:
        p = relu(qk)
    elif activation_enum_int == 6:
        qk = qk.to(v_dtype)
        p = fast_gelu_bf16(qk)
    elif activation_enum_int == 7:
        p = silu(qk)
    elif activation_enum_int == 8:
        p = fast_silu(qk)
    else:
        p = qk

    p *= qk_scale
    p = p.to(v_dtype)
    return p


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def gdpa_kernel_tma_ws_blackwell(
    Q,
    Q_offsets,
    K,
    K_offsets,
    V,
    workspace_ptr,
    Out,  #
    Out_offsets,
    ad_to_request_offset_ptr,
    seq_index,
    stride_qm,
    stride_qh,
    stride_qk,  #
    stride_kn,
    stride_kh,
    stride_kk,  #
    stride_vn,
    stride_vh,
    stride_vk,  #
    stride_om,
    stride_oh,
    stride_ok,  #
    Z,
    H,  # number of q heads.
    G,  # number of q head in each group. number of k v head will be H//G
    N_CTX,
    N_CTX_KV,  #
    qk_scale,  #
    is_predict: tl.constexpr,  #
    Q_SHAPE_0,
    FUSED_QKV: tl.constexpr,  #
    FUSED_KV: tl.constexpr,  #
    SORT_BY_SEQ_LENGTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_D: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    USE_START_END_OFFSETS: tl.constexpr,
    enable_tma: tl.constexpr,
    enable_ws: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
    IS_DENSE_KV: tl.constexpr,
    activation_enum_int: tl.constexpr,
):
    n_tile_num = tl.cdiv(N_CTX, BLOCK_M)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)

    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id

    # start with on-device TMA where descriptors for k, v are set up outside of the persistent
    # loop and descriptor for q is set up inside the persistent loop.
    k_desc = tl.make_tensor_descriptor(
        k,
        shape=[N_CTX_KV * Z, HEAD_DIM * H // G],
        strides=[HEAD_DIM * H // G, 1],
        block_shape=[BLOCK_N, BLOCK_D],
    )
    v_desc = tl.make_tensor_descriptor(
        v,
        shape=[N_CTX_KV * Z, HEAD_DIM * H // G],
        strides=[HEAD_DIM * H // G, 1],
        block_shape=[BLOCK_N, BLOCK_D],
    )

    # allocate buffers for q0, q1
    buffers_q0 = tlx.local_alloc((BLOCK_M // 2, BLOCK_D), tl.float16, 1)
    buffers_q1 = tlx.local_alloc((BLOCK_M // 2, BLOCK_D), tl.float16, 1)
    barrier_q0 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    barrier_q1 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

    # allocate NUM_STAGES buffers for k, v
    buffer_k = tlx.local_alloc((BLOCK_N, BLOCK_D), tl.float16, NUM_STAGES)  # k
    buffer_v = tlx.local_alloc((BLOCK_N, BLOCK_D), tl.float16, NUM_STAGES)  # v

    # allocate tmem for outputs of 4 dots (after partitioning)
    # qk0 = q0 dot k, qk1 = q1 dot k, acc0 = p0 dot v, acc1 = p1 dot v
    result = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem)
    result_2 = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem)
    result_3 = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem)
    result_4 = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem)

    with tlx.async_tasks():
        # activation calculation
        with tlx.async_task("default"):
            for _ in range(0, tiles_per_sm):
                begin_q, qlen = _compute_qlen(tile_idx, n_tile_num, Q_offsets, seq_index, SORT_BY_SEQ_LENGTH)
                if start_m * BLOCK_M < qlen:
                    lo, hi = 0, klen
                    for start_n in range(lo, hi, BLOCK_N):
                        start_n = tl.multiple_of(start_n, BLOCK_N)
                        ## communication channel for qk0, p0
                        _do_activation(qk0_view, qk_scale, v_dtype, activation_enum_int)

        with tlx.async_task(num_warps=4):
            for _ in range(0, tiles_per_sm):
                begin_q, qlen = _compute_qlen(tile_idx, n_tile_num, Q_offsets, seq_index, SORT_BY_SEQ_LENGTH)
                if start_m * BLOCK_M < qlen:
                    lo, hi = 0, klen
                    for start_n in range(lo, hi, BLOCK_N):
                        start_n = tl.multiple_of(start_n, BLOCK_N)
                        ## communication channel for qk1, p1
                        _do_activation(qk1_view, qk_scale, v_dtype, activation_enum_int)

        with tlx.async_task(num_warps=1):  #gemm
            for _ in range(0, tiles_per_sm):
                begin_q, qlen = _compute_qlen(tile_idx, n_tile_num, Q_offsets, seq_index, SORT_BY_SEQ_LENGTH)
                if start_m * BLOCK_M < qlen:
                    _do_dots()
        with tlx.async_task(num_warps=1):  #load
            accum_count_q = 0
            accum_count_k = 0
            for _ in range(0, tiles_per_sm):
                pid = tile_idx % n_tile_num
                off_hz = tile_idx // n_tile_num
                off_z = off_hz // H
                if SORT_BY_SEQ_LENGTH:
                    off_z = tl.load(seq_index + off_z)
                off_q_z = off_z
                off_h = off_hz % H
                off_h_kv = off_h // G

                start_m = pid
                q_offset = off_h.to(tl.int64) * stride_qh
                kv_offset = off_h_kv.to(tl.int64) * stride_kh
                out_offset = off_h.to(tl.int64) * stride_oh

                begin_q, qlen = _compute_qlen(tile_idx, n_tile_num, Q_offsets, seq_index, SORT_BY_SEQ_LENGTH)
                begin_k = tl.load(K_offsets + off_z)
                end_k = tl.load(K_offsets + off_z + 1)
                klen = end_k - begin_k

                if start_m * BLOCK_M < qlen:
                    begin_o = tl.load(Out_offsets + off_z)

                    q_desc = tl.make_tensor_descriptor(
                        q,
                        shape=[end_q.to(tl.int32), HEAD_DIM * H],
                        strides=[HEAD_DIM * H, 1],
                        block_shape=[BLOCK_M // 2, BLOCK_D],
                    )

                    # calculate bufIdx and phase from accum_count_q
                    q_bufIdx = accum_count_q % NUM_BUFFERS_Q
                    q_phase = (accum_count_q // NUM_BUFFERS_Q) & 1
                    _load_tma(q_bufIdx, q_phase, q0_empty_bars, q0_full_bars, buffers_q0, q_desc,
                              begin_q + start_m * BLOCK_M, q_offset, BLOCK_M * BLOCK_D * 2)
                    _load_tma(q_bufIdx, q_phase, q1_empty_bars, q1_full_bars, buffers_q1, q_desc,
                              begin_q + start_m * BLOCK_M + BLOCK_M // 2, q_offset, BLOCK_M * BLOCK_D * 2)
                    lo, hi = 0, klen
                    for start_n in range(lo, hi, BLOCK_N):
                        start_n = tl.multiple_of(start_n, BLOCK_N)
                        k_bufIdx = accum_count_k % NUM_BUFFERS_K
                        k_phase = (accum_count_k // NUM_BUFFERS_K) & 1
                        k_view = _load_k(k_bufIdx, k_phase, k_empty_bars, k_full_bars, buffers_k, k_desc,
                                         begin_k + start_n, kv_offset, BLOCK_N * BLOCK_D * 2)
                        k_view = tlx.local_trans(k_view)

                    accum_count_q += 1

        with tlx.async_task(num_warps=1):  # epilogue
            accum_count_e = 0
            for _ in range(0, tiles_per_sm):
                begin_q, qlen = _compute_qlen(tile_idx, n_tile_num, Q_offsets, seq_index, SORT_BY_SEQ_LENGTH)

                if start_m * BLOCK_M < qlen:
                    o_desc = tl.make_tensor_descriptor(
                        Out,
                        shape=[end_q.to(tl.int32), HEAD_DIM * H],
                        strides=[HEAD_DIM * H, 1],
                        block_shape=[BLOCK_M // 2, BLOCK_D],
                    )

    # allocate NUM_SMEM_BUFFERS buffers for q, k, v
    buffers_k = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float16, NUM_SMEM_BUFFERS)
    buffers_v = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tl.float16, NUM_SMEM_BUFFERS)
    # use multiple TMEM buffers to overlap MMA and epilogue
    tmem_buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, NUM_TMEM_BUFFERS, tlx.storage_kind.tmem)

    # allocate barriers
    smem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    tmem_full_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)
    tmem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)

    with tlx.async_tasks():
        with tlx.async_task("default"):  # producer, TMA load
            # common code duplicated for each region to avoid SMEM overhead
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
            # end of common code

            load_phase = 0  # the current phase of TMA load
            # we virtually "flatten" the two layer loop as if we're performing tma loads on
            # one big list of data
            processed_k_iters = 0
            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_am = pid_m * BLOCK_SIZE_M
                offs_bn = pid_n * BLOCK_SIZE_N

                for k in range(0, k_tiles):
                    # processed_k_iters + k means we use the immediate next buffer slot of tile_id x when we start tile_id x+1
                    buf = (processed_k_iters + k) % NUM_SMEM_BUFFERS
                    a = tlx.local_view(buffers_k, buf)
                    b = tlx.local_view(buffers_v, buf)
                    smem_full_bar = tlx.local_view(smem_full_bars, buf)
                    smem_empty_bar = tlx.local_view(smem_empty_bars, buf)
                    # wait for previous phase(round) of dot for this buf
                    tlx.barrier_wait(smem_empty_bar, load_phase ^ 1)
                    # buffer is now ready to be used again
                    offs_k = k * BLOCK_SIZE_K
                    tlx.barrier_expect_bytes(smem_full_bar, 2 * (BLOCK_SIZE_M + BLOCK_SIZE_N) * BLOCK_SIZE_K)  # float16
                    tlx.async_descriptor_load(a_desc, a, [offs_am, offs_k], smem_full_bar)
                    tlx.async_descriptor_load(b_desc, b, [offs_k, offs_bn], smem_full_bar)
                    # flip phase at the end of a round
                    load_phase = load_phase ^ (buf == NUM_SMEM_BUFFERS - 1)
                processed_k_iters += k_tiles
        with tlx.async_task(num_warps=4, num_regs=232):  # MMA consumer
            # common code duplicated for each region to avoid SMEM overhead
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
            # end of common code

            dot_phase = 0  # the current phase of dot op
            tmem_write_phase = 1  # sync between epilogue consumer and MMA consumer
            cur_tmem_buf = 0

            processed_k_iters = 0
            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_am = pid_m * BLOCK_SIZE_M
                offs_bn = pid_n * BLOCK_SIZE_N

                # init accumulator to 0 (in TMEM), block until the buffer is ready
                zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                acc_tmem = tlx.local_view(tmem_buffers, cur_tmem_buf)
                # wait epilogue consumer to be done with the buffer before reusing it
                tmem_empty_bar = tlx.local_view(tmem_empty_bars, cur_tmem_buf)
                tlx.barrier_wait(tmem_empty_bar, tmem_write_phase)
                # flip phase at the end of a round of using TMEM barriers
                tmem_write_phase = tmem_write_phase ^ (cur_tmem_buf == NUM_TMEM_BUFFERS - 1)

                tlx.local_store(acc_tmem, zeros, tlx.storage_kind.tmem)

                # now iterate along K to compute result for the block
                for k in range(0, k_tiles):
                    # processed_k_iters + k means we use the immediate next buffer slot of tile_id x when we start tile_id x+1
                    buf = (processed_k_iters + k) % NUM_SMEM_BUFFERS
                    a = tlx.local_view(buffers_k, buf)
                    b = tlx.local_view(buffers_v, buf)
                    smem_full_bar = tlx.local_view(smem_full_bars, buf)
                    smem_empty_bar = tlx.local_view(smem_empty_bars, buf)
                    # wait for current phase(round) of load for this buf
                    tlx.barrier_wait(smem_full_bar, dot_phase)
                    # buffer is now ready with loaded data, tlx.async_dot will signal `mBarrier` when done
                    tlx.async_dot(a, b, acc_tmem, mBarriers=[smem_empty_bar], out_dtype=tl.float32)
                    # flip phase at the end of a round
                    dot_phase = dot_phase ^ (buf == NUM_SMEM_BUFFERS - 1)

                # wait for last mma to complete
                last_buf = (processed_k_iters + k_tiles - 1) % NUM_SMEM_BUFFERS
                last_smem_empty_bar = tlx.local_view(smem_empty_bars, last_buf)
                # in case phase was flipped, we should use the phase value when dot op was issued
                last_dot_phase = dot_phase ^ (last_buf == NUM_SMEM_BUFFERS - 1)
                tlx.barrier_wait(last_smem_empty_bar, last_dot_phase)

                # done filling this buffer, signal epilogue consumer
                tmem_full_bar = tlx.local_view(tmem_full_bars, cur_tmem_buf)
                tlx.barrier_arrive(tmem_full_bar, 1)

                # possibly enter next iteration (next tile) without waiting for epilogue
                cur_tmem_buf = (cur_tmem_buf + 1) % NUM_TMEM_BUFFERS
                processed_k_iters += k_tiles

        with tlx.async_task(num_warps=4, num_regs=232):  # epilogue consumer
            # common code duplicated for each region to avoid SMEM overhead
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
            # end of common code

            tmem_read_phase = 0
            cur_tmem_buf = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
                offs_am = pid_m * BLOCK_SIZE_M
                offs_bn = pid_n * BLOCK_SIZE_N

                tmem_full_bar = tlx.local_view(tmem_full_bars, cur_tmem_buf)
                tlx.barrier_wait(tmem_full_bar, tmem_read_phase)
                # flip phase at the end of a round of using TMEM barriers
                tmem_read_phase = tmem_read_phase ^ (cur_tmem_buf == NUM_TMEM_BUFFERS - 1)

                # load the result from TMEM to registers
                acc_tmem = tlx.local_view(tmem_buffers, cur_tmem_buf)

                result = tlx.local_load(acc_tmem, tlx.storage_kind.tmem)
                c = result.to(tl.float16)
                c_desc.store([offs_am, offs_bn], c)

                # done storing this buffer, signal MMA consumer to resume writing to it
                tmem_empty_bar = tlx.local_view(tmem_empty_bars, cur_tmem_buf)
                tlx.barrier_arrive(tmem_empty_bar, 1)

                cur_tmem_buf = (cur_tmem_buf + 1) % NUM_TMEM_BUFFERS


def gdpa(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # Persistent kernel to have thread block resident in SM as long as possible
    grid = lambda META: (min(
        NUM_SMS,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    ), )
    matmul_kernel_tma_ws_blackwell[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        NUM_SMS=NUM_SMS,  #
    )
    return c


torch.manual_seed(0)
M, N, K = 8192, 8192, 8192
a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
torch_output = torch.matmul(a, b)
triton_output = matmul(a, b)
print(f"torch_output_with_fp16_inputs={torch_output}")
print(f"triton_output_with_fp16_inputs={triton_output}")
rtol = 0
torch.testing.assert_close(triton_output, torch_output, atol=1e-2, rtol=rtol)
print("✅ Triton and Torch match")

ref_lib = "cuBLAS"

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=[ref_lib.lower(), "triton"],  # Label name for the lines
        line_names=[ref_lib, "Triton"],  # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-" + ("fp16"),  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))


@triton.testing.perf_report(configs)
def bench_gdpa_attention(BATCH, H, N_CTX, HEAD_DIM, mode, provider, device=DEVICE):
    assert mode in ["fwd"]
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
        fn = lambda: attention(q, k, v, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)

    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    return total_flops * 1e-12 / (ms * 1e-3)


print("Running benchmarks...")
bench_gdpa_attention.run(show_plots=True, print_data=True)
