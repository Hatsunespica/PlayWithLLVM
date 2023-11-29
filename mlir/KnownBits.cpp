#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include <optional>
#include "KnownBits.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#define DEBUG_TYPE "int-range-analysis"

using llvm::KnownBits;
using namespace mlir;
using namespace mlir::dataflow;

KnownBitsRange KnownBitsRange::getMaxRange(Value value) {
    unsigned width =  ConstantIntRanges::getStorageBitwidth(value.getType());
    if (width == 0)
        return {};
    return KnownBitsRange{KnownBits(width)};
}

void KnownBitsRangeLattice::onUpdate(DataFlowSolver *solver) const {
    Lattice::onUpdate(solver);

    // If the integer range can be narrowed to a constant, update the constant
    // value of the SSA value.
    auto value = point.get<Value>();
    auto *cv = solver->getOrCreateState<Lattice<ConstantValue>>(value);
    if (!getValue().getValue().isConstant())
        return solver->propagateIfChanged(
                cv, cv->join(ConstantValue::getUnknownConstant()));

    APInt constant = getValue().getValue().getConstant();
    Dialect *dialect;
    if (auto *parent = value.getDefiningOp())
        dialect = parent->getDialect();
    else
        dialect = value.getParentBlock()->getParentOp()->getDialect();
    solver->propagateIfChanged(
            cv, cv->join(ConstantValue(IntegerAttr::get(value.getType(), constant),
                                       dialect)));
}

#include "transfer.h"

void KnownBitsRangeAnalysis::visitOperation(
        Operation *op, ArrayRef<const KnownBitsRangeLattice *> operands,
        ArrayRef<KnownBitsRangeLattice *> results) {
    // If the lattice on any operand is unitialized, bail out.
    if (llvm::any_of(operands, [](const KnownBitsRangeLattice *lattice) {
        return lattice->getValue().isUninitialized();
    })) {
        return;
    }

    // Ignore non-integer outputs - return early if the op has no scalar
    // integer results
    SmallVector<std::tuple<APInt, APInt>> argRanges(
            llvm::map_range(operands, [](const KnownBitsRangeLattice *val) {
                KnownBits knownBits= val->getValue().getValue();
                return std::make_tuple(knownBits.Zero, knownBits.One);
            }));
    bool hasIntegerResult = false;
    for (auto it : llvm::zip(results, op->getResults())) {
        Value value = std::get<1>(it);
        if (value.getType().isIntOrIndex()) {
            hasIntegerResult = true;
            //unsigned width=IndexType::kInternalStorageBitWidth;
            //if(!value.getType().isIndex()){
            //    width=value.getType().getIntOrFloatBitWidth();
            //}
            std::optional<std::tuple<APInt, APInt>> res;
            //self added
            if (auto castedOp=dyn_cast<mlir::arith::ConstantIntOp>(op);castedOp){
                int64_t value= castedOp.value();
                APInt apInt(castedOp.getOperation()->getResult(0).getType().getIntOrFloatBitWidth(), value);
                llvm::KnownBits knownBits= llvm::KnownBits::makeConstant(apInt);
                res= std::make_tuple(knownBits.Zero, knownBits.One);
            }else if(auto castedOp=dyn_cast<mlir::arith::ConstantIndexOp>(op);castedOp){
                int64_t value= castedOp.value();
                APInt apInt(IndexType::kInternalStorageBitWidth, value);
                llvm::KnownBits knownBits= llvm::KnownBits::makeConstant(apInt);
                res= std::make_tuple(knownBits.Zero, knownBits.One);
            }else{
                res= naiveDispatcher(op, argRanges);
            }

            KnownBitsRange knownBitsRange;
            if(!res.has_value()){
                knownBitsRange=KnownBitsRange::getMaxRange(value);
            }else{
                KnownBits result;
                result.Zero=std::get<0>(*res);
                result.One=std::get<1>(*res);
                knownBitsRange=KnownBitsRange(std::optional<KnownBits>(result));
            }
            //KnownBits knownBits(width);
            //knownBits.setAllOnes();
            //KnownBitsRange knownBitsRange=KnownBitsRange(std::optional<KnownBits>(knownBits));
            ChangeResult changed=results[0]->join(knownBitsRange);
            KnownBitsRangeLattice *lattice = std::get<0>(it);
            propagateIfChanged(lattice, changed);
        } else {
            KnownBitsRangeLattice *lattice = std::get<0>(it);
            propagateIfChanged(lattice,
                               lattice->join(KnownBitsRange::getMaxRange(value)));
        }
    }
    if (!hasIntegerResult)
        return;
    /*llvm::errs()<<"Visit operation\n";
    op->dump();
    llvm::errs()<<llvm::isa<mlir::arith::ConstantOp>(op)<<" "<<(dyn_cast<mlir::arith::ConstantOp>(op)!=nullptr)<<"\n";
    llvm::errs()<<operands.size()<<' '<<results.size()<<"\n";*/
    //setAllToEntryStates(results);
    /*
    auto inferrable = dyn_cast<InferIntRangeInterface>(op);
    if (!inferrable)
        return setAllToEntryStates(results);

    LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");
    SmallVector<KnownBitsRange> argRanges(
            llvm::map_range(operands, [](const KnownBitsRange *val) {
                return val->getValue().getValue();
            }));

    auto joinCallback = [&](Value v, const ConstantIntRanges &attrs) {
        auto result = dyn_cast<OpResult>(v);
        if (!result)
            return;
        assert(llvm::is_contained(op->getResults(), result));

        LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
        KnownBitsRangeLattice *lattice = results[result.getResultNumber()];
        KnownBitsRange oldRange = lattice->getValue();

        ChangeResult changed = lattice->join(KnownBitsRange{attrs});

        // Catch loop results with loop variant bounds and conservatively make
        // them [-inf, inf] so we don't circle around infinitely often (because
        // the dataflow analysis in MLIR doesn't attempt to work out trip counts
        // and often can't).
        bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
            return op->hasTrait<OpTrait::IsTerminator>();
        });
        if (isYieldedResult && !oldRange.isUninitialized() &&
            !(lattice->getValue() == oldRange)) {
            LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
            changed |= lattice->join(IntegerValueRange::getMaxRange(v));
        }
        propagateIfChanged(lattice, changed);
    };

    inferrable.inferResultRanges(argRanges, joinCallback);*/
}

void KnownBitsRangeAnalysis::visitNonControlFlowArguments(
        Operation *op, const RegionSuccessor &successor,
        ArrayRef<KnownBitsRangeLattice *> argLattices, unsigned firstIndex) {
    llvm::errs()<<"Current op\n";
    op->dump();
    /*if (auto inferrable = dyn_cast<InferIntRangeInterface>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");
        // If the lattice on any operand is unitialized, bail out.
        if (llvm::any_of(op->getOperands(), [&](Value value) {
            return getLatticeElementFor(op, value)->getValue().isUninitialized();
        }))
            return;
        SmallVector<ConstantIntRanges> argRanges(
                llvm::map_range(op->getOperands(), [&](Value value) {
                    return getLatticeElementFor(op, value)->getValue().getValue();
                }));

        auto joinCallback = [&](Value v, const ConstantIntRanges &attrs) {
            auto arg = dyn_cast<BlockArgument>(v);
            if (!arg)
                return;
            if (!llvm::is_contained(successor.getSuccessor()->getArguments(), arg))
                return;

            LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
            IntegerValueRangeLattice *lattice = argLattices[arg.getArgNumber()];
            IntegerValueRange oldRange = lattice->getValue();

            ChangeResult changed = lattice->join(IntegerValueRange{attrs});

            // Catch loop results with loop variant bounds and conservatively make
            // them [-inf, inf] so we don't circle around infinitely often (because
            // the dataflow analysis in MLIR doesn't attempt to work out trip counts
            // and often can't).
            bool isYieldedValue = llvm::any_of(v.getUsers(), [](Operation *op) {
                return op->hasTrait<OpTrait::IsTerminator>();
            });
            if (isYieldedValue && !oldRange.isUninitialized() &&
                !(lattice->getValue() == oldRange)) {
                LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
                changed |= lattice->join(IntegerValueRange::getMaxRange(v));
            }
            propagateIfChanged(lattice, changed);
        };

        inferrable.inferResultRanges(argRanges, joinCallback);
        return;
    }*/

    return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
            op, successor, argLattices, firstIndex);
}
