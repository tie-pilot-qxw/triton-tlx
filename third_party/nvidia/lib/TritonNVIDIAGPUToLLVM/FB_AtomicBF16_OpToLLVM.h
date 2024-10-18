/// NOTE: entire contents of this file was copied from
/// //third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp
///
/// Use of this file is tracked by T202302995
///
/// This file is to allow for BF16 atomics via a AtomicCAS loop without the need
/// to write a lot of inline PTX. Reuse of LLVM codegen from AMD allows LLVM
/// to handle the compare and swap loop sequence generation instead of doing
/// it by hand in PTX.

  static LLVM::AtomicOrdering getMemoryOrdering(MemSemantic memOrdering) {
    switch (memOrdering) {
    case MemSemantic::RELAXED:
      return LLVM::AtomicOrdering::monotonic;
    case MemSemantic::ACQUIRE:
      return LLVM::AtomicOrdering::acquire;
    case MemSemantic::RELEASE:
      return LLVM::AtomicOrdering::release;
    case MemSemantic::ACQUIRE_RELEASE:
      return LLVM::AtomicOrdering::acq_rel;
    default:
      return LLVM::AtomicOrdering::acq_rel;
    }
  }

  /// Try to match the mlir::triton::RMWOp to LLVM::AtomicBinOp.
  static std::optional<LLVM::AtomicBinOp> matchAtomicOp(RMWOp atomicOp) {
    switch (atomicOp) {
    case RMWOp::AND:
      return LLVM::AtomicBinOp::_and;
    case RMWOp::OR:
      return LLVM::AtomicBinOp::_or;
    case RMWOp::XOR:
      return LLVM::AtomicBinOp::_xor;
    case RMWOp::ADD:
      return LLVM::AtomicBinOp::add;
    case RMWOp::FADD:
      return LLVM::AtomicBinOp::fadd;
    case RMWOp::MAX:
      return LLVM::AtomicBinOp::max;
    case RMWOp::MIN:
      return LLVM::AtomicBinOp::min;
    case RMWOp::UMAX:
      return LLVM::AtomicBinOp::umax;
    case RMWOp::UMIN:
      return LLVM::AtomicBinOp::umin;
    case RMWOp::XCHG:
      return LLVM::AtomicBinOp::xchg;
    default:
      return std::nullopt;
    }
    llvm_unreachable("Invalid RMWOp");
  }

  LogicalResult
  matchAndRewrite_NOPTX(triton::AtomicRMWOp op, OpAdaptor adaptor,
                        const NVIDIA::TargetInfo &targetInfo,
                        ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();

    auto atomicRmwAttr = op.getAtomicRmwOp();
    Value ptr = op.getPtr();
    Value val = op.getVal();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto valElements = unpackLLElements(loc, llVal, rewriter);
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    Value opResult = op.getResult();
    auto tensorTy = dyn_cast<RankedTensorType>(opResult.getType());
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : opResult.getType();
    const size_t valueElemNbits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(val.getType());
    // vec = 1, numElements = 1 for scalar
    auto vec = getVectorSize(ptr);
    int numElems = 1;
    // tensor
    if (tensorTy) {
      auto valTy = cast<RankedTensorType>(val.getType());
      vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
      // mask
      numElems = tensorTy.getNumElements();
    }
    Value mask = b.int_val(1, 1);
    auto tid = b.tid_val();
    mask = b.and_(mask,
                b.icmp_slt(b.mul(tid, b.i32_val(elemsPerThread)), b.i32_val(numElems)));

    auto memOrdering = op.getSem();
    auto atomicMemOrdering = getMemoryOrdering(memOrdering);

    auto vecTy = vec_ty(valueElemTy, vec);
    auto retType = vec == 1 ? valueElemTy : vecTy;
    SmallVector<Value> resultVals(elemsPerThread);
    const bool f16v2 = vec == 2 && valueElemTy.isF16();
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value rmwPtr = ptrElements[i];
      // TODO: in case llMask is zero we can create only one branch for all
      // elemsPerThread.
      Value rmwMask = llMask ? b.and_(mask, maskElements[i]) : mask;

      Value undefVal = b.undef(retType);
      // Build blocks to bypass the atomic instruction for ~rmwMask.
      auto *curBlock = rewriter.getInsertionBlock();
      auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
      auto *atomicBlock = rewriter.createBlock(
          curBlock->getParent(), std::next(Region::iterator(curBlock)));
      endBlock->addArgument({retType}, {loc});

      rewriter.setInsertionPointToEnd(curBlock);
      rewriter.create<LLVM::CondBrOp>(loc, rmwMask, atomicBlock, endBlock,
                                      undefVal);

      rewriter.setInsertionPointToEnd(atomicBlock);
      auto maybeKind = matchAtomicOp(atomicRmwAttr);
      // TODO: use rocdl.raw.buffer.atomic from ROCDL dialect to use efficient
      // atomics for MI-* series of AMD GPU.
      Value atom = rewriter
                       .create<LLVM::AtomicRMWOp>(
                           loc, *maybeKind, rmwPtr, valElements[i],
                           atomicMemOrdering, StringRef("agent"))
                       .getResult();

      // NV for the f16v2 case generates one packed instruction. We have to
      // create two separate instructions since LLVM::AtomicRMWOp doesn't
      // support this. Can be optimized out with rocdl.raw.buffer.atomic.
      if (f16v2) {
        Value atom2 =
            rewriter
                .create<LLVM::AtomicRMWOp>(
                    loc, *maybeKind, ptrElements[i + 1], valElements[i + 1],
                    atomicMemOrdering, StringRef("agent"))
                .getResult();
        auto tmp = b.insert_element(vecTy, b.undef(vecTy), atom, b.i32_val(0));
        atom = b.insert_element(vecTy, tmp, atom2, b.i32_val(1)).getResult();
      }
      if (!tensorTy) {
        Value atomPtr = getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
        b.store(atom, atomPtr);
      }
      rewriter.create<LLVM::BrOp>(loc, atom, endBlock);

      rewriter.setInsertionPointToStart(endBlock);
      Value retVal = endBlock->getArgument(0);
      if (tensorTy) {
        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? retVal
                       : b.extract_element(valueElemTy, retVal, b.i32_val(ii));
        }
      } else {
        Value atomPtr = getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
        b.barrier();
        Value ret = b.load(valueElemTy, atomPtr);
        b.barrier();
        rewriter.replaceOp(op, {ret});
      }
    }
    if (tensorTy) {
      Type structTy = getTypeConverter()->convertType(tensorTy);
      Value resultStruct = packLLElements(loc, getTypeConverter(), resultVals,
                                          rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }
