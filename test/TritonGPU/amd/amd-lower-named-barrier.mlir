// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm='arch=gfx942' | FileCheck %s

// CHECK: ds_dec_rtn_u32
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @foo1() {
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %0 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x2xi32, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>, #triton_gpu.shared_memory, mutable>

    %c0_i32_0 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32

    %7 = triton_gpu.memdesc_subview %0[%c0_i32, %c1_i32] : !tt.memdesc<1x2xi32, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<1xi32, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>, #triton_gpu.shared_memory, mutable>
    %8 = triton_gpu.memdesc_subview %0[%c0_i32, %c0_i32_0] : !tt.memdesc<1x2xi32, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<1xi32, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>, #triton_gpu.shared_memory, mutable>

    amdgpu.arrive_barrier %8, %7 : !tt.memdesc<1xi32, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>, #triton_gpu.shared_memory, mutable>, !tt.memdesc<1xi32, #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}
