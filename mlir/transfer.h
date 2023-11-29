//
// Created by spica on 7/26/23.
//

#ifndef PLAYWITHLLVM_TRANSFER_H
#define PLAYWITHLLVM_TRANSFER_H

int getConstraint(std::tuple<APInt,APInt> arg0){
    APInt arg0_0=std::get<0>(arg0);
    APInt arg0_1=std::get<1>(arg0);
    APInt andi=arg0_0&arg0_1;
    APInt const0(arg0_0.getBitWidth(),0);
    int result=andi.eq(const0);
    return result;
}

int getInstanceConstraint(std::tuple<APInt,APInt> arg0,APInt inst){
    APInt arg0_0=std::get<0>(arg0);
    APInt arg0_1=std::get<1>(arg0);
    APInt neg_inst=~inst;
    APInt or1=neg_inst|arg0_0;
    APInt or2=inst|arg0_1;
    int cmp1=or1.eq(neg_inst);
    int cmp2=or2.eq(inst);
    int result=cmp1&cmp2;
    return result;
}

APInt OR(APInt arg0,APInt arg1){
    APInt autogen0=arg0|arg1;
    return autogen0;
}

std::tuple<APInt,APInt> ORImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
    APInt arg0_0=std::get<0>(arg0);
    APInt arg0_1=std::get<1>(arg0);
    APInt arg1_0=std::get<0>(arg1);
    APInt arg1_1=std::get<1>(arg1);
    APInt result_0=arg0_0&arg1_0;
    APInt result_1=arg0_1|arg1_1;
    std::tuple<APInt,APInt> result=std::make_tuple(result_0,result_1);
    return result;
}

APInt AND(APInt arg0,APInt arg1){
    APInt autogen1=arg0&arg1;
    return autogen1;
}

std::tuple<APInt,APInt> ANDImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
    APInt arg0_0=std::get<0>(arg0);
    APInt arg0_1=std::get<1>(arg0);
    APInt arg1_0=std::get<0>(arg1);
    APInt arg1_1=std::get<1>(arg1);
    APInt result_0=arg0_0|arg1_0;
    APInt result_1=arg0_1&arg1_1;
    std::tuple<APInt,APInt> result=std::make_tuple(result_0,result_1);
    return result;
}

APInt XOR(APInt arg0,APInt arg1){
    APInt autogen2=arg0^arg1;
    return autogen2;
}

std::tuple<APInt,APInt> XORImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
    APInt arg0_0=std::get<0>(arg0);
    APInt arg0_1=std::get<1>(arg0);
    APInt arg1_0=std::get<0>(arg1);
    APInt arg1_1=std::get<1>(arg1);
    APInt and_00=arg0_0&arg1_0;
    APInt and_11=arg0_1&arg1_1;
    APInt and_01=arg0_0&arg1_1;
    APInt and_10=arg0_1&arg1_0;
    APInt result_0=and_00|and_11;
    APInt result_1=and_01|and_10;
    std::tuple<APInt,APInt> result=std::make_tuple(result_0,result_1);
    return result;
}

APInt getMaxValue(std::tuple<APInt,APInt> arg0){
    APInt arg0_0=std::get<0>(arg0);
    APInt result=~arg0_0;
    return result;
}

APInt getMinValue(std::tuple<APInt,APInt> arg0){
    APInt arg0_1=std::get<1>(arg0);
    return arg0_1;
}

APInt countMinTrailingZeros(std::tuple<APInt,APInt> arg0){
    APInt arg0_0=std::get<0>(arg0);
    unsigned result_autocast=arg0_0.countr_one();
    APInt result(arg0_0.getBitWidth(),result_autocast);
    return result;
}

APInt countMinTrailingOnes(std::tuple<APInt,APInt> arg0){
    APInt arg0_1=std::get<1>(arg0);
    unsigned result_autocast=arg0_1.countr_one();
    APInt result(arg0_1.getBitWidth(),result_autocast);
    return result;
}

std::tuple<APInt,APInt> computeForAddCarry(std::tuple<APInt,APInt> lhs,std::tuple<APInt,APInt> rhs,APInt carryZero,APInt carryOne){
    APInt lhs0=std::get<0>(lhs);
    APInt lhs1=std::get<1>(lhs);
    APInt rhs0=std::get<0>(rhs);
    APInt rhs1=std::get<1>(rhs);
    APInt one(lhs0.getBitWidth(),1);
    APInt negCarryZero=one-carryZero;
    APInt lhsMax=getMaxValue(lhs);
    APInt lhsMin=getMinValue(lhs);
    APInt rhsMax=getMaxValue(rhs);
    APInt rhsMin=getMinValue(rhs);
    APInt possibleSumZeroTmp=lhsMax+rhsMax;
    APInt possibleSumZero=possibleSumZeroTmp+negCarryZero;
    APInt possibleSumOneTmp=lhsMin+rhsMin;
    APInt possibleSumOne=possibleSumOneTmp+carryOne;
    APInt carryKnownZeroTmp0=possibleSumZero^lhs0;
    APInt carryKnownZeroTmp1=carryKnownZeroTmp0^rhs0;
    APInt carryKnownZero=~carryKnownZeroTmp1;
    APInt carryKnownOneTmp=possibleSumOne^lhs1;
    APInt carryKnownOne=carryKnownOneTmp^rhs1;
    APInt lhsKnownUnion=lhs0|lhs1;
    APInt rhsKnownUnion=rhs0|rhs1;
    APInt carryKnownUnion=carryKnownZero|carryKnownOne;
    APInt knownTmp=lhsKnownUnion&rhsKnownUnion;
    APInt known=knownTmp&carryKnownUnion;
    APInt knownZeroTmp=~possibleSumZero;
    APInt knownZero=knownZeroTmp&known;
    APInt knownOne=possibleSumOne&known;
    std::tuple<APInt,APInt> result=std::make_tuple(knownZero,knownOne);
    return result;
}

APInt ADD(APInt arg0,APInt arg1){
    APInt autogen3=arg0+arg1;
    return autogen3;
}

std::tuple<APInt,APInt> ADDImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
    APInt arg1_0=std::get<0>(arg1);
    APInt one(arg1_0.getBitWidth(),1);
    APInt zero(arg1_0.getBitWidth(),0);
    std::tuple<APInt,APInt> result=computeForAddCarry(arg0,arg1,one,zero);
    return result;
}

APInt SUB(APInt arg0,APInt arg1){
    APInt autogen4=arg0-arg1;
    return autogen4;
}

std::tuple<APInt,APInt> SUBImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
    APInt arg1_0=std::get<0>(arg1);
    APInt arg1_1=std::get<1>(arg1);
    std::tuple<APInt,APInt> newRhs=std::make_tuple(arg1_1,arg1_0);
    APInt one(arg1_0.getBitWidth(),1);
    APInt zero(arg1_1.getBitWidth(),0);
    std::tuple<APInt,APInt> result=computeForAddCarry(arg0,newRhs,zero,one);
    return result;
}

APInt MUL(APInt arg0,APInt arg1){
    APInt autogen5=arg0*arg1;
    return autogen5;
}

std::tuple<APInt,APInt> MULImpl(std::tuple<APInt,APInt> arg0,std::tuple<APInt,APInt> arg1){
    APInt arg0Max=getMaxValue(arg0);
    APInt arg1Max=getMaxValue(arg1);
    APInt umaxResult=arg0Max*arg1Max;
    bool umaxResultOverflow;
    arg0Max.umul_ov(arg1Max,umaxResultOverflow);
    APInt zero(arg0Max.getBitWidth(),0);
    unsigned umaxResult_cnt_l_zero_autocast=umaxResult.countl_zero();
    APInt umaxResult_cnt_l_zero(umaxResult.getBitWidth(),umaxResult_cnt_l_zero_autocast);
    APInt leadZ=umaxResultOverflow ? zero : umaxResult_cnt_l_zero ;
    APInt arg0_0=std::get<0>(arg0);
    APInt arg0_1=std::get<1>(arg0);
    APInt arg1_0=std::get<0>(arg1);
    APInt arg1_1=std::get<1>(arg1);
    APInt lhs_union=arg0_0|arg0_1;
    APInt rhs_union=arg1_0|arg1_1;
    unsigned trailBitsKnown0_autocast=lhs_union.countr_one();
    APInt trailBitsKnown0(lhs_union.getBitWidth(),trailBitsKnown0_autocast);
    unsigned trailBitsKnown1_autocast=rhs_union.countr_one();
    APInt trailBitsKnown1(rhs_union.getBitWidth(),trailBitsKnown1_autocast);
    unsigned trailZero0_autocast=arg0_0.countr_one();
    APInt trailZero0(arg0_0.getBitWidth(),trailZero0_autocast);
    unsigned trailZero1_autocast=arg1_0.countr_one();
    APInt trailZero1(arg1_0.getBitWidth(),trailZero1_autocast);
    APInt trailZ=trailZero0+trailZero1;
    APInt smallestOperand_arg0=trailBitsKnown0-trailZero0;
    APInt smallestOperand_arg1=trailBitsKnown1-trailZero1;
    APInt smallestOperand=smallestOperand_arg0.ule(smallestOperand_arg1)?smallestOperand_arg0:smallestOperand_arg1;
    APInt resultBitsKnown_arg0=smallestOperand+trailZ;
    unsigned bitwidth_autocast=arg0_0.getBitWidth();
    APInt bitwidth(arg0_0.getBitWidth(),bitwidth_autocast);
    APInt resultBitsKnown=resultBitsKnown_arg0.ule(bitwidth)?resultBitsKnown_arg0:bitwidth;
    APInt bottomKnown_arg0=arg0_1.getLoBits(trailBitsKnown0.getZExtValue());
    APInt bottomKnown_arg1=arg1_1.getLoBits(trailBitsKnown1.getZExtValue());
    APInt bottomKnown=bottomKnown_arg0*bottomKnown_arg1;
    APInt bottomKnown_neg=~bottomKnown;
    APInt resZerotmp2=bottomKnown_neg.getLoBits(resultBitsKnown.getZExtValue());
    APInt resZerotmp=zero;
    resZerotmp.setHighBits(leadZ.getZExtValue());
    APInt resZero=resZerotmp|resZerotmp2;
    APInt resOne=bottomKnown.getLoBits(resultBitsKnown.getZExtValue());
    std::tuple<APInt,APInt> result=std::make_tuple(resZero,resOne);
    return result;
}

std::optional<std::tuple<APInt,APInt>> naiveDispatcher(Operation* op, ArrayRef<std::tuple<APInt,APInt>> operands){
    if(auto castedOp=dyn_cast<mlir::arith::OrIOp>(op);castedOp){
        return ORImpl(operands[0], operands[1]);
    }
    if(auto castedOp=dyn_cast<mlir::arith::AndIOp>(op);castedOp){
        return ANDImpl(operands[0], operands[1]);
    }
    if(auto castedOp=dyn_cast<mlir::arith::XOrIOp>(op);castedOp){
        return XORImpl(operands[0], operands[1]);
    }
    if(auto castedOp=dyn_cast<mlir::arith::AddIOp>(op);castedOp){
        return ADDImpl(operands[0], operands[1]);
    }
    if(auto castedOp=dyn_cast<mlir::arith::SubIOp>(op);castedOp){
        return SUBImpl(operands[0], operands[1]);
    }
    if(auto castedOp=dyn_cast<mlir::arith::MulIOp>(op);castedOp){
        return MULImpl(operands[0], operands[1]);
    }
    return {};
}



#endif //PLAYWITHLLVM_TRANSFER_H
