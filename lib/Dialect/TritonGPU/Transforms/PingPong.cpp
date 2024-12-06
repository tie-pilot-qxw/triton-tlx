#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <unordered_set>

#define DEBUG_TYPE "triton-ping-pong-sync"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {
namespace triton {
namespace gpu {

// Returns the taskId if op has a single taskId, otherwise, returns -1.
static int getSingleTaskId(Operation *op) {
  if (!op->hasAttr("async_task_id"))
    return -1;
  auto taskArray = op->getAttrOfType<DenseIntElementsAttr>("async_task_id");
  if (taskArray.getValues<int>().size() > 1)
    return -1;
  return taskArray.getValues<int>()[0];
}

// Treat exp2, mulf, addf, reduce as expensive computation when data type is
// a tensor type of 1D or higher.
static bool isExpensiveComp(Operation *op) {
  if (!isa<arith::MulFOp>(op) && !isa<math::Exp2Op>(op) &&
      !isa<arith::AddFOp>(op) && !isa<mlir::triton::ReduceOp>(op))
    return false;
  auto tensorTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  return tensorTy && tensorTy.getRank() >= 1;
}

static Value createGetAsyncTaskId(OpBuilder &builder, Operation *op) {
  auto loc = op->getLoc();
  return builder.create<ttng::GetAsyncTaskIdOp>(loc);
}

#define GEN_PASS_DEF_TRITONGPUPINGPONGSYNC
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUPingPongSyncPass
    : public impl::TritonGPUPingPongSyncBase<TritonGPUPingPongSyncPass> {
public:
  using impl::TritonGPUPingPongSyncBase<
      TritonGPUPingPongSyncPass>::TritonGPUPingPongSyncBase;

  enum class ResourceType {
    Gemm,
    OtherComp,
  };

  void getNestedFor(scf::IfOp ifOp, SmallVector<scf::ForOp> &loops) {
    ifOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  }
  void runOnFuncOp(triton::FuncOp funcOp) {
    // Insert sync points in ForOp for consumer warp groups. Enable this pass
    // when number of consumer warp groups == 2.
    if (numConsumerGroups != 2)
      return;

    SmallVector<scf::ForOp> loops;
    // Identify ForOps for consumer warp groups. Here we assume taskId 0 is for
    // producer. This pass handles the case of a single forOp for two consumer
    // warp groups.
    getOperation()->walk([&](scf::IfOp ifOp) {
      int wgId = getSingleTaskId(ifOp);
      // Assume taskId 0 is for producer.
      if (wgId == 1 || wgId == 2) {
        getNestedFor(ifOp, loops);
      }
    });

    if (!mlir::triton::tools::getBoolEnv("ENABLE_PINGPONG"))
      return;
    if (loops.size() != 1)
      return;

    Operation *startOfGemm = nullptr;
    Operation *endOfGemm = nullptr;
    // FIXME: only handle the first loop.
    auto forOp = loops[0];
    OpBuilder builder(forOp);
    // A simple heuristic for now:
    //   Mark the start of a gemm section when hitting a DotLike op.
    //   Mark the end of a gemm section once hitting a expensive cuda op.
    for (auto &op : forOp.getBody()->without_terminator()) {
      if (startOfGemm && endOfGemm)
        break;
      bool isCudaCore = isExpensiveComp(&op);
      if (op.hasTrait<OpTrait::DotLike>() && !isCudaCore &&
          startOfGemm == nullptr) {
        startOfGemm = &op;
        continue;
      }
      if (!op.hasTrait<OpTrait::DotLike>() && isCudaCore && startOfGemm) {
        endOfGemm = &op;
        break;
      }
    }
    if (!startOfGemm || !endOfGemm)
      return;

    LLVM_DEBUG({
      LDBG("found start of tensor core ops");
      startOfGemm->dump();
    });
    LLVM_DEBUG({
      LDBG("found end of tensor core ops");
      endOfGemm->dump();
    });

    // FIXME: hard-code using named barrier 9 and 10 in this pass.
    // Prior to the forOp, add "bar.arrive 9, 256" only when task Id is 2.
    // At startOfGemm, insert "bar.sync 8+taskId, 256"
    // At endOfGemm, insert "bar.arrive 11-taskId, 256"
    builder.setInsertionPoint(forOp);
    auto forLoc = forOp->getLoc();

    // FIXME: hard-code total number of threads to be 256 when numConsumerGroups
    // is 2.
    Value numThreads = builder.create<arith::ConstantIntOp>(forLoc, 256, 32);
    Value c_9 = builder.create<arith::ConstantIntOp>(forLoc, 9, 32);

    // "bar.arrive 9, 256" only when task Id is 2.
    Value c_2 = builder.create<arith::ConstantIntOp>(forLoc, 2, 32);
    Value curTaskId = createGetAsyncTaskId(builder, forOp);
    auto pred = builder.create<arith::CmpIOp>(forLoc, arith::CmpIPredicate::eq,
                                              curTaskId, c_2);
    auto ifOp = builder.create<scf::IfOp>(forLoc, pred, /*else=*/false);
    builder.setInsertionPoint(ifOp.thenYield());
    builder.create<ttng::NamedBarrierArriveOp>(forLoc, c_9, numThreads);

    // At startOfGemm, insert "bar.sync 8+taskId, 256"
    // 8 + taskId: 9 for taskId 1 and 10 for taskId 2.
    builder.setInsertionPoint(startOfGemm);
    auto loc = startOfGemm->getLoc();
    Value c_8 = builder.create<arith::ConstantIntOp>(loc, 8, 32);
    Value syncBarrier = builder.create<arith::AddIOp>(loc, c_8, curTaskId);
    builder.create<ttng::NamedBarrierWaitOp>(loc, syncBarrier, numThreads);

    // At endOfGemm, insert "bar.arrive 11-taskId, 256"
    // 11 - taskId: 10 for taskId 1 and 9 for taskId2.
    builder.setInsertionPoint(endOfGemm);
    auto loc2 = endOfGemm->getLoc();
    Value c_11 = builder.create<arith::ConstantIntOp>(loc2, 11, 32);
    Value arriveBarrier = builder.create<arith::SubIOp>(loc2, c_11, curTaskId);
    builder.create<ttng::NamedBarrierArriveOp>(loc2, arriveBarrier, numThreads);
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
    LLVM_DEBUG({
      LDBG("post pass");
      getOperation()->dump();
    });
    return;
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
