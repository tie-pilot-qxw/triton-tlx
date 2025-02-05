#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

using namespace mlir;

namespace {

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                    mlir::gpu::Dimension::y,
                                                    mlir::gpu::Dimension::z};
    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);
    Value blockId =
        rewriter.create<::mlir::gpu::GridDimOp>(loc, dims[op.getAxisAsInt()]);
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, i32_ty, blockId);
    return success();
  }
};

struct ArriveBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::ArriveBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::ArriveBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto countSmemObj = LLVM::getSharedMemoryObjectFromStruct(
      op.getLoc(), adaptor.getCount(),
      typeConverter->convertType(op.getCount().getType().getElementType()),
      rewriter);

    auto phaseSmemObj = LLVM::getSharedMemoryObjectFromStruct(
      op.getLoc(), adaptor.getPhase(),
      typeConverter->convertType(op.getPhase().getType().getElementType()),
      rewriter);
    // TBD - pass wraparound value through the ttgir OP
    // set wraparound to WAVES_PER_TASK - 1 (3)
    auto wrapAroundVal = LLVM::createConstantI32(loc, rewriter, 3);


    GCNBuilder gcnBuilder;
    auto &dec_rtn = *gcnBuilder.create("ds_dec_rtn_u32");
    auto retVal = gcnBuilder.newOperand("=v");
    auto countBaseAddr = gcnBuilder.newOperand(countSmemObj.getBase(), "v");
    auto wav = gcnBuilder.newOperand(wrapAroundVal, "v");
    dec_rtn(retVal, countBaseAddr, wav);
    auto &wait_cnt = *gcnBuilder.create("s_waitcnt lgkmcnt(0)");
    wait_cnt();
    auto res = gcnBuilder.launch(rewriter, loc, i32_ty, true /*hasSideEffects*/);

    Value zero = i32_val(0);
    Value allArrived = icmp_eq(res, zero);

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterPhaseFlipBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *phaseFlipBlock = rewriter.createBlock(afterPhaseFlipBlock);
    rewriter.setInsertionPointToEnd(currentBlock);

    rewriter.create<LLVM::CondBrOp>(loc, allArrived , phaseFlipBlock,
                                    afterPhaseFlipBlock);

    MLIRContext *ctx = rewriter.getContext();
    rewriter.setInsertionPointToStart(phaseFlipBlock);
    auto phaseBaseAddr = phaseSmemObj.getBase();
    GCNBuilder gcnBuilder1;
    Value one = i32_val(1);
    auto &xor_phase = *gcnBuilder1.create("ds_xor_b32");
    auto baseAddrArg = gcnBuilder1.newOperand(phaseBaseAddr, "v");
    auto oneArg = gcnBuilder1.newOperand(one, "v");
    xor_phase(baseAddrArg, oneArg);
    // wake up sleeping threads
    // threads waiting onother barrier may also be spuriously woken up
    auto &s_wakeup = *gcnBuilder1.create("s_wakeup");
    s_wakeup();
    gcnBuilder1.launch(rewriter, loc, void_ty(ctx), true /*hasSideEffects*/);

    auto br = rewriter.create<LLVM::BrOp>(loc, afterPhaseFlipBlock);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct InitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::InitBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::InitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto countSmemObj = LLVM::getSharedMemoryObjectFromStruct(
      op.getLoc(), adaptor.getCountAlloc(),
      typeConverter->convertType(op.getCountAlloc().getType().getElementType()),
      rewriter);
    int count = op.getCount();
    auto countBaseAddr = countSmemObj.getBase();
    Value countVal = i32_val(count);
    GCNBuilder countStoreBuilder;
    auto &count_store = *countStoreBuilder.create("ds_write_b32");
    auto baseAddrArg = countStoreBuilder.newOperand(countBaseAddr, "v");
    auto countArg = countStoreBuilder.newOperand(countVal, "v");
    count_store(baseAddrArg, countArg);
    countStoreBuilder.launch(rewriter, loc, void_ty(ctx), true /*hasSideEffects*/);

    auto phaseSmemObj = LLVM::getSharedMemoryObjectFromStruct(
      op.getLoc(), adaptor.getPhaseAlloc(),
      typeConverter->convertType(op.getPhaseAlloc().getType().getElementType()),
      rewriter);
    int phase = op.getPhase();
    auto phaseBaseAddr = phaseSmemObj.getBase();
    Value phaseVal = i32_val(phase);
    GCNBuilder phaseStoreBuilder;
    auto &phase_store = *phaseStoreBuilder.create("ds_write_b32");
    auto phaseAddrArg = phaseStoreBuilder.newOperand(phaseBaseAddr, "v");
    auto phaseArg = phaseStoreBuilder.newOperand(phaseVal, "v");
    phase_store(phaseAddrArg, phaseArg);
    auto &wait_cnt = *phaseStoreBuilder.create("s_waitcnt lgkmcnt(0)");
    wait_cnt();
    phaseStoreBuilder.launch(rewriter, loc, void_ty(ctx), true /*hasSideEffects*/);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReadBarrierPhaseOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::ReadBarrierPhaseOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::ReadBarrierPhaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto phaseSmemObj = LLVM::getSharedMemoryObjectFromStruct(
      op.getLoc(), adaptor.getPhaseAlloc(),
      typeConverter->convertType(op.getPhaseAlloc().getType().getElementType()),
      rewriter);
    auto phaseBaseAddr = phaseSmemObj.getBase();
    GCNBuilder phaseReadBuilder;
    auto &phase_read = *phaseReadBuilder.create("ds_read_b32");
    auto retVal = phaseReadBuilder.newOperand("=v");
    auto phaseAddrArg = phaseReadBuilder.newOperand(phaseBaseAddr, "v");
    phase_read(retVal, phaseAddrArg);
    auto &wait_cnt = *phaseReadBuilder.create("s_waitcnt lgkmcnt(0)");
    wait_cnt();
    auto res = phaseReadBuilder.launch(rewriter, loc, i32_ty, true /*hasSideEffects*/);
    rewriter.replaceOp(op, res);
    return success();
   }
};


} // namespace  

void mlir::triton::AMD::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
  patterns.add<ArriveBarrierOpConversion>(typeConverter, benefit);
  patterns.add<InitBarrierOpConversion>(typeConverter, benefit);
  patterns.add<ReadBarrierPhaseOpConversion>(typeConverter, benefit);
}
