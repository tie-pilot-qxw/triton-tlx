#include "CodePartitionUtility.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "nvgpu-ws-memory-planner"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

using OperationListT = std::vector<Operation *>;

static Channel *findChannelForAlloc(Value value,
                                    SmallVector<Channel *> &channels) {
  Operation *op = value.getDefiningOp();
  Channel *TheCh = nullptr;
  for (auto *ch : channels) {
    Operation *alloc = ch->getAllocOp();
    if (alloc == op) {
      TheCh = ch;
      break;
    }
  }
  return TheCh;
}

static void updateLiveOpsInOneBlock(Channel *TheCh, OperationListT &liveOps) {
  assert(TheCh->channelKind == DataChannelKind::TMEMPost ||
         TheCh->channelKind == DataChannelKind::SMEMPost);
  Operation *src = TheCh->getSrcOp();
  SmallVector<Operation *> dsts;
  TheCh->getDstOps(dsts);
  Operation *lastDst = TheCh->getDstOpLast();
  // Assuming they are in the same block, insert ops from src to dsts.
  auto *block = src->getBlock();
  bool foundStart = false;
  for (auto &op : block->getOperations()) {
    if (&op == src) {
      foundStart = true;
      liveOps.push_back(&op);
      continue;
    }
    if (foundStart)
      liveOps.push_back(&op);
    if (&op == lastDst) {
      break;
    }
  }
}

namespace triton {
// A simplified version of AllocationAnalysis.
class MemoryPlanner {
public:
  MemoryPlanner(Operation *operation, Allocation *allocation,
                SmallVector<Channel *> *channels)
      : operation(operation), allocation(allocation), channels(channels) {}

private:
  using BufferT = Allocation::BufferT;
  using BufferRangeMapT = llvm::MapVector<BufferT *, Interval<size_t>>;
  Operation *operation;
  Allocation *allocation;
  SmallVector<Channel *> *channels;
  BufferRangeMapT bufferRange;

  void getExplicitValueSize(Operation *op) {
    auto alloc = dyn_cast<ttg::LocalAllocOp>(op);
    if (!alloc || !alloc.isSharedMemoryAlloc())
      return;
    auto allocType = alloc.getType();
    int64_t numElems = 0;
    if (auto paddedEnc =
            dyn_cast<ttg::PaddedSharedEncodingAttr>(allocType.getEncoding())) {
      SmallVector<int64_t> unpaddedShape = ttg::getShapePerCTA(allocType);
      numElems = paddedEnc.getPaddedSize(unpaddedShape);
    } else {
      auto shapePerCTA = ttg::getAllocationShapePerCTA(allocType);
      numElems = product<int64_t>(shapePerCTA);
    }
    int64_t bytes = numElems * allocType.getElementTypeBitWidth() / 8;

    auto alignment = alloc.getAlignmentOrDefault();
    allocation->addBuffer<BufferT::BufferKind::Explicit>(alloc, bytes,
                                                         alignment);
  }

  void getValuesAndSizes() {
    // Get the alloc values
    operation->walk<WalkOrder::PreOrder>(
        [&](Operation *op) { getExplicitValueSize(op); });
  }

  void resolveExplicitBufferLiveness(
      function_ref<Interval<size_t>(Value value)> getLiveness) {
    for (auto valueBufferIter : allocation->valueBuffer) {
      auto value = valueBufferIter.first;
      auto *buffer = valueBufferIter.second;
      bufferRange[buffer] = getLiveness(value);
      LLVM_DEBUG({
        llvm::dbgs() << "-- buffer " << buffer->id << "; value: ";
        value.dump();
      });
    }
  }

  OperationListT livenessForSmemChannel(Value value) {
    // Find the channel for value in channels.
    ChannelPost *TheCh =
        static_cast<ChannelPost *>(findChannelForAlloc(value, *channels));
    std::vector<Operation *> liveOps;
    updateLiveOpsInOneBlock(TheCh, liveOps);
    return liveOps;
  }

  void resolveLiveness() {
    DenseMap<Operation *, size_t> operationId;
    operation->walk<WalkOrder::PostOrder>(
        [&](Operation *op) { operationId[op] = operationId.size(); });

    // Analyze liveness of explicit buffers
    Liveness liveness(operation);
    auto getValueLivenessRange = [&](Value value) {
      Operation *defOp = value.getDefiningOp();
      LLVM_DEBUG({
        llvm::dbgs() << "-- getValueLivenessRange \n";
        value.dump();
      });
      auto liveOperations = livenessForSmemChannel(value);
      auto minId = std::numeric_limits<size_t>::max();
      auto maxId = std::numeric_limits<size_t>::min();
      llvm::for_each(liveOperations, [&](Operation *liveOp) {
        LLVM_DEBUG(llvm::dbgs()
                   << "---- liveOp " << operationId[liveOp] << "\n");
        if (defOp && isa<mlir::triton::gpu::WarpSpecializeOp>(defOp)) {
          minId = 0;
          maxId = operationId.size();
          return;
        }
        if (operationId[liveOp] < minId) {
          minId = operationId[liveOp];
        }
        if ((operationId[liveOp] + 1) > maxId) {
          maxId = operationId[liveOp] + 1;
        }
      });
      return Interval(minId, maxId);
    };

    resolveExplicitBufferLiveness(getValueLivenessRange);
  }

public:
  void run() {
    getValuesAndSizes();
    resolveLiveness();
    // Try to set buffer.copy, heuristics: for channels in innermost loop, set
    // to maxStage Make sure the configuration will fit in SMEM.
  }
  void dumpBuffers() const {
    LDBG("Dump bufferRange: id size offset ---------");
    for (auto bufferIter : bufferRange) {
      llvm::dbgs() << "-- " << bufferIter.first->id << " "
                   << bufferIter.first->size << " " << bufferIter.first->offset;
      llvm::dbgs() << " interval " << bufferIter.second.start() << " "
                   << bufferIter.second.end() << "\n";
    }
  }
};
} // namespace triton

static void handleOperandD(ttng::TMEMAllocOp tmemAllocOp,
                           std::vector<Operation *> &liveOps) {
  DenseSet<Operation *> users;
  DenseSet<Operation *> userScopes; // users in the same scope
  bool first = true;
  for (auto user : tmemAllocOp.getResult().getUsers()) {
    users.insert(user);
    if (first) {
      userScopes.insert(user);
    } else {
      // We may need to lift the scopes in userScopes.
      auto *scope = *(userScopes.begin());
      auto *sameLevel = getSameLevelOp(user, scope);
      if (sameLevel->getBlock() == user->getBlock()) {
        // user stays unchanged, scope gets lifted to sameLevel.
        userScopes.clear();
        userScopes.insert(sameLevel);
      } else {
        // scope stays unchanged.
        userScopes.insert(sameLevel);
      }
    }
    first = false;
  }
  // Find the block that contains all users
  bool foundStart = false;
  auto *scope = *(userScopes.begin());
  Operation *lastDst = nullptr;
  for (auto &op : scope->getBlock()->getOperations()) {
    if (userScopes.count(&op)) {
      lastDst = &op;
    }
  }
  for (auto &op : scope->getBlock()->getOperations()) {
    if (userScopes.count(&op) || foundStart) {
      foundStart = true;
      // Goes through nested regions.
      op.walk<WalkOrder::PostOrder>(
          [&](Operation *nestedOp) { liveOps.push_back(nestedOp); });
    }
    if (&op == lastDst) {
      break;
    }
  }
}

// Return the list of operations where value is live.
OperationListT livenessForTmemChannel(Value value,
                                      SmallVector<Channel *> &channels) {
  // Find the channel for value in channels.
  ttng::TmemDataChannelPost *TheCh = static_cast<ttng::TmemDataChannelPost *>(
      findChannelForAlloc(value, channels));
  std::vector<Operation *> liveOps;
  // Operand D can be associated with multiple channels. From first producer to
  // last consumer.
  if (TheCh->isOperandD) {
    handleOperandD(cast<ttng::TMEMAllocOp>(TheCh->getAllocOp()), liveOps);
  } else {
    // Assume a single producer, multiple consumers all under the same block.
    updateLiveOpsInOneBlock(TheCh, liveOps);
  }
  return liveOps;
}

// Copied from TensorMemoryAllocation.cpp
static Interval<int> getLiveIntervals(Value value, Liveness &liveness,
                                      DenseMap<Operation *, int> &operationId,
                                      SmallVector<Channel *> &channels) {
  auto liveOperations = livenessForTmemChannel(value, channels);
  // Merge the alloc liverange with the liverange of any subview of the
  // allocation.
  SmallVector<Operation *> users(value.getUsers());
  while (!users.empty()) {
    Operation *user = users.pop_back_val();
    if (!isa<ttg::MemDescIndexOp, ttg::MemDescReinterpretOp>(user))
      continue;
    auto usersLivness = livenessForTmemChannel(user->getResult(0), channels);
    liveOperations.insert(liveOperations.end(), usersLivness.begin(),
                          usersLivness.end());
    users.append(user->getResult(0).getUsers().begin(),
                 user->getResult(0).getUsers().end());
  }
  auto minId = std::numeric_limits<int>::max();
  auto maxId = std::numeric_limits<int>::min();
  std::for_each(liveOperations.begin(), liveOperations.end(),
                [&](Operation *liveOp) {
                  if (operationId[liveOp] < minId) {
                    minId = operationId[liveOp];
                  }
                  if ((operationId[liveOp] + 1) > maxId) {
                    maxId = operationId[liveOp] + 1;
                  }
                });
  return Interval(minId, maxId);
}

class RowIdConstraints {
  llvm::EquivalenceClasses<Operation *> dependentAllocs;
  llvm::SmallDenseMap<Operation *, int> rowIndex;

public:
  void joinOps(Operation *op1, Operation *op2) {
    dependentAllocs.unionSets(op1, op2);
  }

  std::optional<int> getRowIdConstraint(Operation *op) {
    auto it = dependentAllocs.findLeader(op);
    if (it == dependentAllocs.member_end())
      return std::nullopt;
    auto rowIt = rowIndex.find(*it);
    if (rowIt == rowIndex.end())
      return std::nullopt;
    return rowIt->second;
  }

  void addConstraints(Operation *op, int rowId) {
    auto it = dependentAllocs.findLeader(op);
    if (it == dependentAllocs.member_end())
      return;
    rowIndex[*it] = rowId;
  }
};

static void allocateTMem(Operation *parentOp,
                         SmallVector<Channel *> &channels) {
  SmallVector<triton::nvidia_gpu::TMEMAllocOp> allocs;
  DenseMap<Operation *, int> operationId;
  RowIdConstraints rowIdConstraints;
  // Only consider allocs for channels.
  parentOp->walk<WalkOrder::PostOrder>([&](Operation *op) {
    operationId[op] = operationId.size();
    if (auto alloc = dyn_cast<triton::nvidia_gpu::TMEMAllocOp>(op)) {
      allocs.push_back(alloc);
    }
  });
  Liveness liveness(parentOp);
  DenseMap<Operation *, Interval<int>> allocToIntervals;
  for (auto it = allocs.begin(), e = allocs.end(); it != e; ++it) {
    ttng::TMEMAllocOp alloc = *it;
    Interval<int> liveInterval =
        getLiveIntervals(alloc, liveness, operationId, channels);
    auto memDescType = alloc.getType();
    ttng::TMemAllocation allocSize = ttng::getTmemAllocSizes(memDescType);
    LLVM_DEBUG(alloc.dump());
    LDBG("tmem livenss: " << liveInterval.start() << " " << liveInterval.end());
    LDBG("tmem allocSize: " << allocSize.numCols << " " << allocSize.numRows);

    ttng::TmemDataChannelPost *TheCh = static_cast<ttng::TmemDataChannelPost *>(
        findChannelForAlloc(alloc, channels));
    // Should we extend live range for isOperandD?
    allocToIntervals[alloc.getOperation()] = liveInterval;
  }
  int totalMemorySize = ttng::allocateTMemWithInterval(allocToIntervals);
  LDBG("tmem size: " << totalMemorySize);
}

void doMemoryPlanner(triton::FuncOp &funcOp, unsigned numBuffers) {

  // Step 1: collect all communications between producers and consumers.
  SmallVector<std::unique_ptr<Channel>> channelsOrigin;
  collectPostChannels(channelsOrigin, funcOp);
  SmallVector<Channel *> channels;
  for (const auto &c : channelsOrigin) {
    channels.push_back(c.get());
  }
  if (channels.empty()) {
    return;
  }
  // Step 2: figure out smem/tmem sizes and liveness.
  // If two buffers are sharing a multi-staged alloc, the liveness can overlap,
  // otherwise, the liveness can't overlap.
  Allocation allocation;
  triton::MemoryPlanner planner(funcOp, &allocation, &channels);
  planner.run();
  LLVM_DEBUG(planner.dumpBuffers());
  allocateTMem(funcOp, channels);
  // Emit buffer reuse decisions based on live ranges. Use heuristics to choose
  // a single configuration for now. Make sure it fits.
  // Start with the biggest alloc. Do not reuse channel with isOperandD true.
}

#define GEN_PASS_DEF_NVGPUTESTWSMEMORYPLANNER
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestWSMemoryPlannerPass
    : public impl::NVGPUTestWSMemoryPlannerBase<NVGPUTestWSMemoryPlannerPass> {
public:
  using impl::NVGPUTestWSMemoryPlannerBase<
      NVGPUTestWSMemoryPlannerPass>::NVGPUTestWSMemoryPlannerBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    if (numBuffers >= 1)
      doMemoryPlanner(funcOp, numBuffers);
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
