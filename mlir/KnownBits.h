#ifndef PLAYWITHLLVM_KNOWNBITS_H
#define PLAYWITHLLVM_KNOWNBITS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/Support/KnownBits.h"

using llvm::KnownBits;
using namespace mlir;
using namespace mlir::dataflow;

/// This lattice value represents the integer range of an SSA value.
class KnownBitsRange {
public:
    /// Create a maximal range ([0, uint_max(t)] / [int_min(t), int_max(t)])
    /// range that is used to mark the value as unable to be analyzed further,
    /// where `t` is the type of `value`.
    static KnownBitsRange getMaxRange(Value value);

    /// Create an integer value range lattice value.
    KnownBitsRange(std::optional<KnownBits> value = std::nullopt)
            : value(std::move(value)) {}

    /// Whether the range is uninitialized. This happens when the state hasn't
    /// been set during the analysis.
    bool isUninitialized() const { return !value.has_value(); }

    /// Get the known integer value range.
    const KnownBits &getValue() const {
        assert(!isUninitialized());
        return *value;
    }

    /// Compare two ranges.
    bool operator==(const KnownBitsRange &rhs) const {
        return value == rhs.value;
    }

    /// Take the union of two ranges.
    static KnownBitsRange join(const KnownBitsRange &lhs,
                               const KnownBitsRange &rhs) {
        if (lhs.isUninitialized())
            return rhs;
        if (rhs.isUninitialized())
            return lhs;
        return KnownBitsRange{lhs.getValue().unionWith(rhs.getValue())};
    }

    /// Print the integer value range.
    void print(raw_ostream &os) const { os << value; }

private:
    /// The known integer value range.
    std::optional<KnownBits> value;
};

/// This lattice element represents the integer value range of an SSA value.
/// When this lattice is updated, it automatically updates the constant value
/// of the SSA value (if the range can be narrowed to one).
class KnownBitsRangeLattice : public Lattice<KnownBitsRange> {
public:
    using Lattice::Lattice;

    /// If the range can be narrowed to an integer constant, update the constant
    /// value of the SSA value.
    void onUpdate(DataFlowSolver *solver) const override;
};

/// Integer range analysis determines the integer value range of SSA values
/// using operations that define `InferIntRangeInterface` and also sets the
/// range of iteration indices of loops with known bounds.
class KnownBitsRangeAnalysis
        : public SparseForwardDataFlowAnalysis<KnownBitsRangeLattice> {
public:
    using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

    /// At an entry point, we cannot reason about interger value ranges.
    void setToEntryState(KnownBitsRangeLattice *lattice) override {
        propagateIfChanged(lattice, lattice->join(KnownBitsRange::getMaxRange(
                lattice->getPoint())));
    }

    /// Visit an operation. Invoke the transfer function on each operation that
    /// implements `InferIntRangeInterface`.
    void visitOperation(Operation *op,
                        ArrayRef<const KnownBitsRangeLattice *> operands,
                        ArrayRef<KnownBitsRangeLattice *> results) override;

    /// Visit block arguments or operation results of an operation with region
    /// control-flow for which values are not defined by region control-flow. This
    /// function calls `InferIntRangeInterface` to provide values for block
    /// arguments or tries to reduce the range on loop induction variables with
    /// known bounds.
    void
    visitNonControlFlowArguments(Operation *op, const RegionSuccessor &successor,
                                 ArrayRef<KnownBitsRangeLattice *> argLattices,
                                 unsigned firstIndex) override;
};

#endif //PLAYWITHLLVM_KNOWNBITS_H
