import torch

import triton
import triton.language as tl
from triton.tools.experimental_descriptor import (
    create_1d_tma_descriptor_type,
    create_2d_tma_descriptor_type,
)


def map_dtype_to_triton(dtype: torch.dtype) -> int:
    """
    Maps torch dtype to triton dtype.
    Args:
        dtype (torch.dtype): input dtype.
    Returns:
        tl.dtype: triton dtype.
    """
    if dtype == torch.float16:
        return 0
    elif dtype == torch.bfloat16:
        return 1
    elif dtype == torch.float32:
        return 2
    elif dtype == torch.int32:
        return 3
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


####################################################################################################
# Vector Reduce-add with on-host TMA
####################################################################################################


@triton.jit
def vector_add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    x_desc,
    y_ptr,  # *Pointer* to second input vector.
    y_desc,
    output_desc,
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    # Load x through TMA.
    x = tl._experimental_descriptor_load(
        x_desc, [block_start], [BLOCK_SIZE], x_ptr.dtype.element_ty
    )
    # Store x to through TMA.
    tl._experimental_descriptor_store(output_desc, x, [block_start])
    # Load y through TMA.
    y = tl._experimental_descriptor_load(
        y_desc, [block_start], [BLOCK_SIZE], y_ptr.dtype.element_ty
    )
    tl.debug_barrier()
    # Store y to through TMA reduce add.
    tl._experimental_descriptor_store(output_desc, y, [block_start], store_reduce="add")


def vector_add(x: torch.Tensor, y: torch.Tensor):
    BLOCK_SIZE = 256
    x_desc = create_1d_tma_descriptor_type(
        x.data_ptr(), size, BLOCK_SIZE, map_dtype_to_triton(x.dtype)
    )
    y_desc = create_1d_tma_descriptor_type(
        y.data_ptr(), size, BLOCK_SIZE, map_dtype_to_triton(y.dtype)
    )
    output = torch.empty_like(x)
    output_desc = create_1d_tma_descriptor_type(
        output.data_ptr(), size, BLOCK_SIZE, map_dtype_to_triton(output.dtype)
    )
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](x, x_desc, y, y_desc, output_desc, BLOCK_SIZE=BLOCK_SIZE)
    return output


torch.manual_seed(0)
size = 98432
dtype = torch.float32
x = torch.rand(size, dtype=dtype, device="cuda")
y = torch.rand(size, dtype=dtype, device="cuda")
output_torch = x + y
output_triton = vector_add(x, y)
print(output_torch)
print(output_triton)
print(
    f"The maximum difference between torch and triton is "
    f"{torch.max(torch.abs(output_torch - output_triton))}"
)
assert torch.equal(output_torch, output_triton)

# %%
# Benchmark
# ---------
#
# We can now benchmark our custom op on vectors of increasing sizes to get a sense of how it does relative to PyTorch.
# To make things easier, Triton has a set of built-in utilities that allow us to concisely plot the performance of our custom ops.
# for different problem sizes.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**i for i in range(12, 28, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch"],  # Possible values for `line_arg`.
        line_names=["Triton", "Torch"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name=f"vector-add-{dtype}-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device="cuda", dtype=dtype)
    y = torch.rand(size, device="cuda", dtype=dtype)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vector_add(x, y), quantiles=quantiles
        )
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# # %%
# # We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# # `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)

####################################################################################################
# Tile Reduce-add with on-host TMA
####################################################################################################


@triton.jit
def tile_add_kernel(
    x_ptr,
    x_desc,
    y_ptr,
    y_desc,
    z_ptr,
    z_desc,
    output_desc,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    GROUP_SIZE: tl.constexpr,
):
    BLOCK_SIZE_M: tl.constexpr = BLOCK_SIZE
    BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE
    GROUP_SIZE_M: tl.constexpr = GROUP_SIZE
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M
    offs_n = pid_n * BLOCK_SIZE_N

    # Load x through TMA.
    x = tl._experimental_descriptor_load(
        x_desc, [offs_m, offs_n], [BLOCK_SIZE, BLOCK_SIZE], x_ptr.dtype.element_ty
    )
    # Store x to through TMA.
    tl._experimental_descriptor_store(output_desc, x, [offs_m, offs_n])
    # Load y through TMA.
    y = tl._experimental_descriptor_load(
        y_desc, [offs_m, offs_n], [BLOCK_SIZE, BLOCK_SIZE], y_ptr.dtype.element_ty
    )
    tl.debug_barrier()
    # Store y to through TMA reduce add.
    tl._experimental_descriptor_store(
        output_desc, y, [offs_m, offs_n], store_reduce="add"
    )
    z = tl._experimental_descriptor_load(
        z_desc, [offs_m, offs_n], [BLOCK_SIZE, BLOCK_SIZE], z_ptr.dtype.element_ty
    )
    tl.debug_barrier()
    tl._experimental_descriptor_store(
        output_desc, z, [offs_m, offs_n], store_reduce="add"
    )


def tile_add(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    BLOCK_SIZE = 64
    M, N = x.shape
    dtype = x.dtype
    x_desc = create_2d_tma_descriptor_type(
        x.data_ptr(), M, N, BLOCK_SIZE, BLOCK_SIZE, map_dtype_to_triton(x.dtype)
    )
    y_desc = create_2d_tma_descriptor_type(
        y.data_ptr(), M, N, BLOCK_SIZE, BLOCK_SIZE, map_dtype_to_triton(y.dtype)
    )
    z_desc = create_2d_tma_descriptor_type(
        z.data_ptr(), M, N, BLOCK_SIZE, BLOCK_SIZE, map_dtype_to_triton(z.dtype)
    )
    output = torch.empty((M, N), device=x.device, dtype=dtype)
    output_desc = triton.tools.experimental_descriptor.create_2d_tma_descriptor_type(
        output.data_ptr(), M, N, BLOCK_SIZE, BLOCK_SIZE, map_dtype_to_triton(x.dtype)
    )

    assert x.is_cuda and y.is_cuda and output.is_cuda
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]) * triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    tile_add_kernel[grid](
        x,
        x_desc,
        y,
        y_desc,
        z,
        z_desc,
        output_desc,
        M,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=8,
    )
    return output


torch.manual_seed(0)
size = 512
dtype = torch.bfloat16
x = torch.rand((size, size), dtype=dtype, device="cuda")
y = torch.rand((size, size), dtype=dtype, device="cuda")
z = torch.rand((size, size), dtype=dtype, device="cuda")
output_torch = x + y + z
output_triton = tile_add(x, y, z)
print(output_torch)
print(output_triton)

if torch.allclose(output_triton, output_torch, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        # x_vals=[
        #     2**i for i in range(12, 15, 1)
        # ],  # Different possible values for `x_name`.
        x_vals=[
            128 * i for i in range(2, 33)
        ],  # Different possible values for `x_name`
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch"],  # Possible values for `line_arg`.
        line_names=["Triton", "Torch"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name=f"tile-add-{dtype}-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand((size, size), device="cuda", dtype=dtype)
    y = torch.rand((size, size), device="cuda", dtype=dtype)
    z = torch.rand((size, size), device="cuda", dtype=dtype)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x + y + z, quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: tile_add(x, y, z), quantiles=quantiles
        )
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# # %%
# # We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# # `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)

####################################################################################################
# Tile Reduce-add with on-device TMA
####################################################################################################


@triton.jit
def add_kernel_tma_reduce(
    workspace_ptr,
    x_ptr,  # *Pointer* to first input tensor.
    y_ptr,  # *Pointer* to second input tensor.
    z_ptr,  # *Pointer* to third input tensor.
    output_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    GROUP_SIZE: tl.constexpr,
):
    BLOCK_SIZE_M: tl.constexpr = BLOCK_SIZE
    BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE
    GROUP_SIZE_M: tl.constexpr = GROUP_SIZE
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M
    offs_n = pid_n * BLOCK_SIZE_N

    TMA_SIZE: tl.constexpr = 128
    workspace_base = workspace_ptr + pid * 4 * TMA_SIZE
    x_desc_ptr = workspace_base
    y_desc_ptr = workspace_base + TMA_SIZE
    z_desc_ptr = workspace_base + 2 * TMA_SIZE
    output_desc_ptr = workspace_base + 3 * TMA_SIZE

    tl.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=x_desc_ptr,
        global_address=x_ptr,
        load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        global_size=[M, N],
        element_ty=x_ptr.dtype.element_ty,
    )
    tl.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=y_desc_ptr,
        global_address=y_ptr,
        load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        global_size=[M, N],
        element_ty=y_ptr.dtype.element_ty,
    )
    tl.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=z_desc_ptr,
        global_address=z_ptr,
        load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        global_size=[M, N],
        element_ty=z_ptr.dtype.element_ty,
    )
    tl.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=output_desc_ptr,
        global_address=output_ptr,
        load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        global_size=[M, N],
        element_ty=output_ptr.dtype.element_ty,
    )

    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(x_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(y_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(z_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(output_desc_ptr)

    # Load x through TMA.
    x = tl._experimental_descriptor_load(
        x_desc_ptr, [offs_m, offs_n], [BLOCK_SIZE, BLOCK_SIZE], x_ptr.dtype.element_ty
    )
    # Store x to through TMA.
    tl.debug_barrier()
    tl._experimental_descriptor_store(output_desc_ptr, x, [offs_m, offs_n])
    # Load y through TMA.
    y = tl._experimental_descriptor_load(
        y_desc_ptr, [offs_m, offs_n], [BLOCK_SIZE, BLOCK_SIZE], y_ptr.dtype.element_ty
    )
    # Store y to through TMA reduce add.
    tl.debug_barrier()
    tl._experimental_descriptor_store(
        output_desc_ptr, y, [offs_m, offs_n], store_reduce="add"
    )
    # Load z through TMA.
    z = tl._experimental_descriptor_load(
        z_desc_ptr, [offs_m, offs_n], [BLOCK_SIZE, BLOCK_SIZE], z_ptr.dtype.element_ty
    )
    # Store y to through TMA reduce add.
    tl.debug_barrier()
    tl._experimental_descriptor_store(
        output_desc_ptr, z, [offs_m, offs_n], store_reduce="add"
    )


@triton.jit
def add_kernel_tma_store(
    workspace_ptr,
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    z_ptr,  # *Pointer* to third input vector.
    output_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    GROUP_SIZE: tl.constexpr,
):
    BLOCK_SIZE_M: tl.constexpr = BLOCK_SIZE
    BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE
    GROUP_SIZE_M: tl.constexpr = GROUP_SIZE
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M
    offs_n = pid_n * BLOCK_SIZE_N

    TMA_SIZE: tl.constexpr = 128
    workspace_base = workspace_ptr + pid * 4 * TMA_SIZE
    x_desc_ptr = workspace_base
    y_desc_ptr = workspace_base + TMA_SIZE
    z_desc_ptr = workspace_base + 2 * TMA_SIZE
    output_desc_ptr = workspace_base + 3 * TMA_SIZE

    tl.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=x_desc_ptr,
        global_address=x_ptr,
        load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        global_size=[M, N],
        element_ty=x_ptr.dtype.element_ty,
    )
    tl.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=y_desc_ptr,
        global_address=y_ptr,
        load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        global_size=[M, N],
        element_ty=y_ptr.dtype.element_ty,
    )
    tl.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=z_desc_ptr,
        global_address=z_ptr,
        load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        global_size=[M, N],
        element_ty=z_ptr.dtype.element_ty,
    )
    tl.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=output_desc_ptr,
        global_address=output_ptr,
        load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        global_size=[M, N],
        element_ty=output_ptr.dtype.element_ty,
    )

    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(x_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(y_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(z_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(output_desc_ptr)

    # Load x through TMA.
    x = tl._experimental_descriptor_load(
        x_desc_ptr, [offs_m, offs_n], [BLOCK_SIZE, BLOCK_SIZE], x_ptr.dtype.element_ty
    )
    # Store x to through TMA.
    tl._experimental_descriptor_store(output_desc_ptr, x, [offs_m, offs_n])
    # Load x through TMA.
    x_stored = tl._experimental_descriptor_load(
        output_desc_ptr,
        [offs_m, offs_n],
        [BLOCK_SIZE, BLOCK_SIZE],
        x_ptr.dtype.element_ty,
    )
    # Load y through TMA.
    y = tl._experimental_descriptor_load(
        y_desc_ptr, [offs_m, offs_n], [BLOCK_SIZE, BLOCK_SIZE], y_ptr.dtype.element_ty
    )
    # Store x + y to through TMA.
    tl.debug_barrier()
    tl._experimental_descriptor_store(output_desc_ptr, x_stored + y, [offs_m, offs_n])
    # Load x + y through TMA.
    x_plus_y = tl._experimental_descriptor_load(
        output_desc_ptr,
        [offs_m, offs_n],
        [BLOCK_SIZE, BLOCK_SIZE],
        y_ptr.dtype.element_ty,
    )
    # Load z through TMA.
    z = tl._experimental_descriptor_load(
        z_desc_ptr, [offs_m, offs_n], [BLOCK_SIZE, BLOCK_SIZE], z_ptr.dtype.element_ty
    )
    # Store y to through TMA reduce add.
    tl.debug_barrier()
    tl._experimental_descriptor_store(output_desc_ptr, x_plus_y + z, [offs_m, offs_n])


def add_tma_reduce(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    BLOCK_SIZE = 64
    M, N = x.shape
    dtype = x.dtype

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    tma_size = 128
    workspace = torch.empty(NUM_SMS * 4 * tma_size, dtype=torch.uint8, device="cuda")

    output = torch.zeros((M, N), device=x.device, dtype=dtype)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]) * triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    add_kernel_tma_reduce[grid](
        workspace,
        x,
        y,
        z,
        output,
        M,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=8,
    )
    return output


def add_tma_store(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    BLOCK_SIZE = 64
    M, N = x.shape
    dtype = x.dtype

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    tma_size = 128
    workspace = torch.empty(NUM_SMS * 4 * tma_size, dtype=torch.uint8, device="cuda")

    output = torch.zeros((M, N), device=x.device, dtype=dtype)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]) * triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    add_kernel_tma_store[grid](
        workspace,
        x,
        y,
        z,
        output,
        M,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=8,
    )
    return output


torch.manual_seed(0)
size = 512
dtype = torch.float32
x = torch.rand((size, size), dtype=dtype, device="cuda")
y = torch.rand((size, size), dtype=dtype, device="cuda")
z = torch.rand((size, size), dtype=dtype, device="cuda")

output_torch = x + y + z
output_triton_tma_reduce = add_tma_reduce(x, y, z)
output_triton_tma_store = add_tma_store(x, y, z)
print(output_torch)
print(output_triton_tma_reduce)
print(output_triton_tma_store)

if torch.allclose(
    output_triton_tma_reduce, output_torch, atol=1e-2, rtol=0
) and torch.allclose(output_triton_tma_store, output_torch, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        # x_vals=[
        #     2**i for i in range(12, 15, 1)
        # ],  # Different possible values for `x_name`.
        x_vals=[
            128 * i for i in range(2, 33)
        ],  # Different possible values for `x_name`
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=[
            "triton-tma-store",
            "triton-tma-reduce",
            "torch",
        ],  # Possible values for `line_arg`.
        line_names=[
            "Triton-TMA-store",
            "Triton-TMA-reduce",
            "Torch",
        ],  # Label name for the lines.
        styles=[("blue", "-"), ("blue", "--"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name=f"tile-add-{dtype}-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand((size, size), device="cuda", dtype=dtype)
    y = torch.rand((size, size), device="cuda", dtype=dtype)
    z = torch.rand((size, size), device="cuda", dtype=dtype)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x + y + z, quantiles=quantiles
        )
    if provider == "triton-tma-store":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add_tma_store(x, y, z), quantiles=quantiles
        )
    if provider == "triton-tma-reduce":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add_tma_reduce(x, y, z), quantiles=quantiles
        )
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# # %%
# # We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# # `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)
