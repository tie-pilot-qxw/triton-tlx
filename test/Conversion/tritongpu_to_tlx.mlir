// RUN: triton-opt -split-input-file %s --tritongpu-print-tlx | FileCheck %s

// Generated from:
// @triton.jit
// def add_vecs_kernel(
//     x_ptr,
//     y_ptr,
//     z_ptr,
//     n_elements,
//     BLOCK_SIZE: tl.constexpr,
// ):
//     pid = tl.program_id(axis=0)
//     block_start = pid * BLOCK_SIZE
//     offsets = block_start + tl.arange(0, BLOCK_SIZE)
//     mask = offsets < n_elements
//     x = tl.load(x_ptr + offsets, mask=mask)
//     y = tl.load(y_ptr + offsets, mask=mask)
//     tl.store(z_ptr + offsets, x + y, mask=mask)
//
// src = triton.compiler.ASTSource(
//     fn=add_vecs_kernel,
//     constexprs={'BLOCK_SIZE': 1024},
//     signature={'x_ptr': "*fp32", 'y_ptr': "*fp32", 'z_ptr': "*fp32", "n_elements": "i32",'BLOCK_SIZE': 'constexpr'},
// )
// triton.compile(src=src, target=triton.backends.compiler.GPUTarget("cuda", 100, 32))

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_vecs_kernel(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %z_ptr: !tt.ptr<f32>, %n_elements: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c1024_i32 : i32
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %offsets_0 = tt.splat %block_start : i32 -> tensor<1024xi32, #blocked>
    %offsets_1 = arith.addi %offsets_0, %offsets : tensor<1024xi32, #blocked>
    %mask = tt.splat %n_elements : i32 -> tensor<1024xi32, #blocked>
    %mask_2 = arith.cmpi slt, %offsets_1, %mask : tensor<1024xi32, #blocked>
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_3 = tt.addptr %x, %offsets_1 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x_4 = tt.load %x_3, %mask_2 : tensor<1024x!tt.ptr<f32>, #blocked>
    %y = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %y_5 = tt.addptr %y, %offsets_1 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %y_6 = tt.load %y_5, %mask_2 : tensor<1024x!tt.ptr<f32>, #blocked>
    %0 = tt.splat %z_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %1 = tt.addptr %0, %offsets_1 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %2 = arith.addf %x_4, %y_6 : tensor<1024xf32, #blocked>
    tt.store %1, %2, %mask_2 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK:      @triton.jit
// CHECK-NEXT: def add_vecs_kernel_0(v_0, v_1, v_2, v_3):
// CHECK-NEXT:   v_4 = 1024
// CHECK-NEXT:   v_5 = tl.program_id(0)
// CHECK-NEXT:   v_6 = v_5 * v_4
// CHECK-NEXT:   v_7 = tl.arange(0, 1024)
// CHECK-NEXT:   v_8 = tl.full((1024, ), v_6, tl.int32)
// CHECK-NEXT:   v_9 = v_8 + v_7
// CHECK-NEXT:   v_10 = tl.full((1024, ), v_3, tl.int32)
// CHECK-NEXT:   v_11 = v_9 < v_10
// CHECK-NEXT:   v_12 = tl.full((1024, ), v_0, tl.pointer_type(tl.float32))
// CHECK-NEXT:   v_13 = v_12 + v_9
// CHECK-NEXT:   v_14 = tl.load(v_13, v_11)
// CHECK-NEXT:   v_15 = tl.full((1024, ), v_1, tl.pointer_type(tl.float32))
// CHECK-NEXT:   v_16 = v_15 + v_9
// CHECK-NEXT:   v_17 = tl.load(v_16, v_11)
// CHECK-NEXT:   v_18 = tl.full((1024, ), v_2, tl.pointer_type(tl.float32))
// CHECK-NEXT:   v_19 = v_18 + v_9
// CHECK-NEXT:   v_20 = v_14 + v_17
// CHECK-NEXT:   tl.store(v_19, v_20, v_11)
// CHECK-NEXT:   return ()

// -----

// Generated from:
// @triton.jit
// def add(a, b, c, d):
//     return (a + b, c + d)
//
// @triton.jit
// def partial_sum_kernel(
//     x_ptr,
//     y_ptr,
//     z_ptr,
//     n_elements,
//     BLOCK_SIZE: tl.constexpr,
// ):
//     pid = tl.program_id(axis=0)
//     block_start = pid * BLOCK_SIZE
//     offsets = block_start + tl.arange(0, BLOCK_SIZE)
//     mask = offsets < n_elements
//     x = tl.load(x_ptr + offsets, mask=mask)
//     y = tl.load(y_ptr + offsets, mask=mask)
//     val = tl.reduce((x, y), 0, add)
//     tl.store(z_ptr + pid, val[0] + val[1])
//
// src = triton.compiler.ASTSource(
//     fn=partial_sum_kernel,
//     constexprs={'BLOCK_SIZE': 1024},
//     signature={'x_ptr': "*fp32", 'y_ptr': "*fp32", 'z_ptr': "*fp32", "n_elements": "i32",'BLOCK_SIZE': 'constexpr'},
// )
// triton.compile(src=src, target=triton.backends.compiler.GPUTarget("cuda", 100, 32))

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @partial_sum_kernel(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %z_ptr: !tt.ptr<f32>, %n_elements: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c1024_i32 : i32
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %offsets_0 = tt.splat %block_start : i32 -> tensor<1024xi32, #blocked>
    %offsets_1 = arith.addi %offsets_0, %offsets : tensor<1024xi32, #blocked>
    %mask = tt.splat %n_elements : i32 -> tensor<1024xi32, #blocked>
    %mask_2 = arith.cmpi slt, %offsets_1, %mask : tensor<1024xi32, #blocked>
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_3 = tt.addptr %x, %offsets_1 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x_4 = tt.load %x_3, %mask_2 : tensor<1024x!tt.ptr<f32>, #blocked>
    %y = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %y_5 = tt.addptr %y, %offsets_1 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %y_6 = tt.load %y_5, %mask_2 : tensor<1024x!tt.ptr<f32>, #blocked>
    %val:2 = "tt.reduce"(%x_4, %y_6) <{axis = 0 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32):
      %val_7 = arith.addf %arg4, %arg5 : f32
      %val_8 = arith.addf %arg6, %arg7 : f32
      tt.reduce.return %val_7, %val_8 : f32, f32
    }) : (tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>) -> (f32, f32)
    %0 = tt.addptr %z_ptr, %pid : !tt.ptr<f32>, i32
    %1 = arith.addf %val#0, %val#1 : f32
    tt.store %0, %1 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK:      @triton.jit
// CHECK-NEXT: def reduce_fn_1(v_20, v_21, v_22, v_23):
// CHECK-NEXT:   v_24 = v_20 + v_21
// CHECK-NEXT:   v_25 = v_22 + v_23
// CHECK-NEXT:   return (v_24, v_25, )
// CHECK-NEXT: @triton.jit
// CHECK-NEXT: def partial_sum_kernel_0(v_0, v_1, v_2, v_3):
// CHECK-NEXT:   v_4 = 1024
// CHECK-NEXT:   v_5 = tl.program_id(0)
// CHECK-NEXT:   v_6 = v_5 * v_4
// CHECK-NEXT:   v_7 = tl.arange(0, 1024)
// CHECK-NEXT:   v_8 = tl.full((1024, ), v_6, tl.int32)
// CHECK-NEXT:   v_9 = v_8 + v_7
// CHECK-NEXT:   v_10 = tl.full((1024, ), v_3, tl.int32)
// CHECK-NEXT:   v_11 = v_9 < v_10
// CHECK-NEXT:   v_12 = tl.full((1024, ), v_0, tl.pointer_type(tl.float32))
// CHECK-NEXT:   v_13 = v_12 + v_9
// CHECK-NEXT:   v_14 = tl.load(v_13, v_11)
// CHECK-NEXT:   v_15 = tl.full((1024, ), v_1, tl.pointer_type(tl.float32))
// CHECK-NEXT:   v_16 = v_15 + v_9
// CHECK-NEXT:   v_17 = tl.load(v_16, v_11)
// CHECK-NEXT:   (v_18, v_19, ) = tl.reduce((v_14, v_17, ), 0, reduce_fn_1)
// CHECK-NEXT:   v_26 = v_2 + v_5
// CHECK-NEXT:   v_27 = v_18 + v_19
// CHECK-NEXT:   tl.store(v_26, v_27)
// CHECK-NEXT:   return ()
