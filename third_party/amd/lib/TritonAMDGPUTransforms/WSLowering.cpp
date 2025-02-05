#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include <set>

#include "mlir/IR/OperationSupport.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace {

static const int THREADS_PER_WAVE = 64;
static const int WAVES_PER_TASK = 4;

void lowerGetAsyncTaskIdOp(Operation *parentOp) {
  DenseSet<Operation *> eraseOps;
  parentOp->walk([&](ttng::GetAsyncTaskIdOp op) {
    auto loc = op.getLoc();
    OpBuilder builder(op);
    auto i32ty = builder.getIntegerType(32);
    auto workIDX = builder.create<ROCDL::ThreadIdXOp>(loc, i32ty);
    auto constWaveSize = builder.create<arith::ConstantIntOp>(loc, THREADS_PER_WAVE * WAVES_PER_TASK, 32);
    auto warpIDX = builder.create<arith::DivSIOp>(loc, workIDX, constWaveSize);
    op.getResult().replaceAllUsesWith(warpIDX);
    eraseOps.insert(op);
  });
  for (Operation *op : eraseOps)
    op->erase();
}

void lowerInit(OpBuilder thenBuilder, Location loc, Value barrierCountView,
               Value barrierPhaseView, int initCount, int phase,
               Value threadId) {
  thenBuilder.create<triton::amdgpu::InitBarrierOp>(loc, barrierCountView, initCount, barrierPhaseView, phase);
}

Value getMBarrierPhaseBit(OpBuilder &builder, Operation *op,
                          bool emptyBarrier) {
  auto loc = op->getLoc();
  assert(isa<ttng::ProducerAcquireOp>(op) || isa<ttng::ConsumerWaitOp>(op));
  Value curPhase;
  if (auto acq = dyn_cast<ttng::ProducerAcquireOp>(op))
    curPhase = acq.getPhase();
  else if (auto wait = dyn_cast<ttng::ConsumerWaitOp>(op))
    curPhase = wait.getPhase();
  if (emptyBarrier) {
    Value _1_1b = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    curPhase = builder.create<mlir::arith::XOrIOp>(loc, curPhase, _1_1b);
  }
  return curPhase;
}

void processAcquireOpOrWaitOp(OpBuilder &builder, Operation *op,
                              Value barrierPhaseView, bool isProducer) {
  Value localPhase = getMBarrierPhaseBit(builder, op, isProducer);
  auto loc = op->getLoc();
  auto i32Ty = builder.getIntegerType(32);
  localPhase = builder.create<arith::ExtUIOp>(loc, i32Ty, localPhase);

  auto initialCondition = localPhase;
  auto whileOp = builder.create<scf::WhileOp>(loc, TypeRange{}, ValueRange{});
  // before block
  Block *beforeBlock = builder.createBlock(&whileOp.getBefore());
  builder.setInsertionPointToEnd(beforeBlock);

  Value barrierPhase = builder.create<triton::amdgpu::ReadBarrierPhaseOp>(loc, i32Ty, barrierPhaseView);
  Value phaseCond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  barrierPhase, localPhase);
  builder.create<scf::ConditionOp>(loc, phaseCond, ValueRange{});
  
  // after block
  Block *afterBlock = builder.createBlock(&whileOp.getAfter());
  builder.setInsertionPointToEnd(afterBlock);
  auto SLEEP_TIME = builder.create<arith::ConstantIntOp>(loc, 10, 32);
  auto sleepInstrinsic = "llvm.amdgcn.s.sleep";
  auto sleepOp = LLVM::createLLVMIntrinsicCallOp(builder, loc, sleepInstrinsic, TypeRange{}, ValueRange{SLEEP_TIME});
  builder.create<scf::YieldOp>(loc,  ValueRange{});
  // wake up sleeping threads
  // "llvm.amdgcn.s.wakeup" not implemented yet
  builder.setInsertionPointAfter(whileOp);

  auto asmDialectAttr = LLVM::AsmDialectAttr::get(builder.getContext(),
                                                      LLVM::AsmDialect::AD_ATT);
  const char *asmStr = "s_wakeup";
  const char *constraints = "";
  builder.create<LLVM::InlineAsmOp>(
    loc,
    /*resultTypes=*/TypeRange(), /*operands=*/ValueRange(),
    /*asm_string=*/asmStr, constraints, /*has_side_effects=*/false,
    /*is_align_stack=*/false,
    /*asm_dialect=*/asmDialectAttr,
    /*operand_attrs=*/ArrayAttr());

}

void processCommitOpOrReleaseOp(OpBuilder &builder, Operation *op, Value bufferCountView, Value bufferPhaseView, Value threadId) {
  auto loc = op->getLoc();
  auto threadsPerWave = builder.create<arith::ConstantIntOp>(loc, THREADS_PER_WAVE, 32); 
  auto mod = builder.create<arith::RemSIOp>(loc, threadId, threadsPerWave);
  auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  auto cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, mod, zero);
  auto ifOp = builder.create<scf::IfOp>(loc, cond);
  auto thenBuilder = ifOp.getThenBodyBuilder();
  thenBuilder.create<triton::amdgpu::ArriveBarrierOp>(loc, bufferCountView, bufferPhaseView);
}

void lowerTokenOperations(Operation *parentOp) {
  SmallVector<Operation *> eraseOps;
  parentOp->walk([&](ttng::CreateTokenOp createTokenOp) {
    MLIRContext *context = createTokenOp.getContext();
    OpBuilder builder(createTokenOp);
    Location loc = createTokenOp.getLoc();

    Attribute sharedMemorySpace =
        triton::gpu::SharedMemorySpaceAttr::get(context);
    auto barrierCTALayout =
        ttg::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                                /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto barrierEncoding =
        ttg::SharedEncodingAttr::get(context, 1, 1, 1, {0}, barrierCTALayout);

    Type barrierMemDescType =
        tt::MemDescType::get({createTokenOp.getNum(), 2}, builder.getI32Type(),
                             barrierEncoding, sharedMemorySpace,
                             /*mutableMemory=*/true);

    Type barrierElementMemDescType =
        tt::MemDescType::get({1}, builder.getI32Type(), barrierEncoding,
                             sharedMemorySpace, /*mutableMemory=*/true);

    Value bufferFullArray = builder.create<mlir::triton::gpu::LocalAllocOp>(
        loc, barrierMemDescType, Value());
    Value bufferEmptyArray = builder.create<mlir::triton::gpu::LocalAllocOp>(
        loc, barrierMemDescType, Value());

    auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    auto one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    // Helper function for extracting one element
    auto createFieldView = [&](OpBuilder builder, Location loc, Value array,
                               Value barrierIndexOp,
                               Value fieldIndex) -> Value {
      SmallVector<Value> elementIdx({barrierIndexOp, fieldIndex});
      return builder.create<ttg::MemDescSubviewOp>(
          loc, barrierElementMemDescType, array, elementIdx);
    };

    // Initialize the barriers
    // If thread0, set barrier to initCount
    auto i32ty = builder.getIntegerType(32);
    auto threadId = builder.create<ROCDL::ThreadIdXOp>(loc, i32ty);
    Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               threadId, zero);
    auto ifOp = builder.create<scf::IfOp>(loc, cond);
    auto thenBuilder = ifOp.getThenBodyBuilder();
    Value phaseInitValue = zero;
    Value phaseOffset = one;
    Value countOffset = zero;

    for (unsigned barrierIndex = 0; barrierIndex < createTokenOp.getNum();
         barrierIndex++) {
      Value barrierIndexOp =
          thenBuilder.create<arith::ConstantIntOp>(loc, barrierIndex, 32);
      Value bufferFullCountView = createFieldView(
          thenBuilder, loc, bufferFullArray, barrierIndexOp, countOffset);
      Value bufferFullPhaseView = createFieldView(
          thenBuilder, loc, bufferFullArray, barrierIndexOp, phaseOffset);
      lowerInit(thenBuilder, loc, bufferFullCountView, bufferFullPhaseView,
                WAVES_PER_TASK - 1, 0, threadId);

      Value bufferEmptyCountView = createFieldView(
          thenBuilder, loc, bufferEmptyArray, barrierIndexOp, countOffset);
      Value bufferEmptyPhaseView = createFieldView(
          thenBuilder, loc, bufferEmptyArray, barrierIndexOp, phaseOffset);
      lowerInit(thenBuilder, loc, bufferEmptyCountView, bufferEmptyPhaseView,
                WAVES_PER_TASK - 1, 0, threadId);
    }
    // sbarrier to ensure named barrier initialization is done
    builder.create<mlir::gpu::BarrierOp>(loc);

    for (Operation *user : createTokenOp.getResult().getUsers()) {
      auto loc = user->getLoc();
      builder.setInsertionPoint(user);
      if (auto op = dyn_cast<ttng::ProducerAcquireOp>(user)) {
        Value bufferEmptyPhaseView = createFieldView(builder, loc, bufferEmptyArray,
                                                 op.getIdx(), phaseOffset);
        processAcquireOpOrWaitOp(builder, op, bufferEmptyPhaseView, true);
      } else if (auto op = dyn_cast<ttng::ConsumerWaitOp>(user)) {
        Value bufferFullPhaseView = createFieldView(builder, loc, bufferFullArray,
                                                 op.getIdx(), phaseOffset);
        processAcquireOpOrWaitOp(builder, op, bufferFullPhaseView, false);
      } else if (auto op = dyn_cast<ttng::ProducerCommitOp>(user)) {
        Value bufferFullPhaseView = createFieldView(builder, loc, bufferFullArray,
                                                 op.getIdx(), phaseOffset);
        Value bufferFullCountView = createFieldView(builder, loc, bufferFullArray,
                                                 op.getIdx(), countOffset);
        processCommitOpOrReleaseOp(builder, op, bufferFullCountView, bufferFullPhaseView, threadId);
      } else if (auto op = dyn_cast<ttng::ConsumerReleaseOp>(user)) {
        Value bufferEmptyPhaseView = createFieldView(builder, loc, bufferEmptyArray,
                                                 op.getIdx(), phaseOffset);
        Value bufferEmptyCountView = createFieldView(builder, loc, bufferEmptyArray,
                                                 op.getIdx(), countOffset);
        processCommitOpOrReleaseOp(builder, op, bufferEmptyCountView, bufferEmptyPhaseView, threadId);
      }
      eraseOps.push_back(user);
    }
    eraseOps.push_back(createTokenOp);
  });

  for (Operation *op : eraseOps)
    op->erase();
}
} // namespace


class TritonAMDGPUWSLoweringPass
    : public TritonAMDGPUWSLoweringBase<TritonAMDGPUWSLoweringPass> {
public:
  TritonAMDGPUWSLoweringPass() = default;
  TritonAMDGPUWSLoweringPass(int numConsumerGroups) { this->numConsumerGroups = numConsumerGroups; }
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    lowerGetAsyncTaskIdOp(mod);
    lowerTokenOperations(mod);
    auto builder = OpBuilder::atBlockBegin(mod.getBody());
    mod->setAttr("triton_gpu.num-warp-groups-per-cta",
                 builder.getI32IntegerAttr(1 + numConsumerGroups));
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUWSLoweringPass(int numConsumerGroups) {
  return std::make_unique<TritonAMDGPUWSLoweringPass>(numConsumerGroups);
}
