//===- Hello.cpp - Example code from "Writing an LLVM Pass" ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements two versions of the LLVM "Hello World" pass described
// in docs/WritingAnLLVMPass.html
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Constants.h"
#include <vector>
using namespace llvm;

#define DEBUG_TYPE "hello"

STATISTIC(HelloCounter, "Counts number of functions greeted");

namespace {
  // Hello - The first implementation, without getAnalysisUsage.
  struct Hello : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    Hello() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override {
      ++HelloCounter;
      errs() << "Hello: ";
      errs().write_escaped(F.getName()) << '\n';
      return false;
    }
  };
}

char Hello::ID = 0;
static RegisterPass<Hello> X("hello", "Hello World Pass");

namespace {
  // Hello2 - The second implementation with getAnalysisUsage implemented.
  struct Hello2 : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    Hello2() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override {
      /*++HelloCounter;
      errs() << "Hello: ";
      errs().write_escaped(F.getName()) << '\n';*/
      unsigned int basicBlockCount=0;
      unsigned int instructionCount=0;
      for(BasicBlock& bb:F){
        ++basicBlockCount;
        for(Instruction&i :bb){
            ++instructionCount;
        }
      }
      errs()<<"Hello2: ";
      errs().write_escaped(F.getName())<<"Basic Block:"<<basicBlockCount<<"\nInsturction"
        <<instructionCount<<"\n";
      return false;
    }

    // We don't modify the program, so we preserve all analyses.
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
    }
  };
}

char Hello2::ID = 0;
static RegisterPass<Hello2>
Y("hello2", "Hello World Pass (with getAnalysisUsage implemented)");

namespace {
  // Hello2 - The second implementation with getAnalysisUsage implemented.
  struct Hello3 : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    Hello3() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override {
      /*++HelloCounter;
      errs() << "Hello: ";
      errs().write_escaped(F.getName()) << '\n';
      unsigned int basicBlockCount=0;
      unsigned int instructionCount=0;
      for(BasicBlock& bb:F){
        ++basicBlockCount;
        for(Instruction&i :bb){
            ++instructionCount;
        }
      }
      errs()<<"Hello2: ";
      errs().write_escaped(F.getName())<<"Basic Block:"<<basicBlockCount<<"\nInsturction"
        <<instructionCount<<"\n";*/
      for(BasicBlock& bb:F){
        for(Instruction& i:bb){
            CallSite cs(&i);
            if(!cs.getInstruction()){
                continue;
            }
            Value *called=cs.getCalledValue()->stripPointerCasts();
            if(Function* f=dyn_cast<Function>(called)){
                errs()<<"\tDirect call to function:"<<f->getName()<<" from "<<F.getName()<<"\n";
            }
        }
      }
      return false;
    }

    // We don't modify the program, so we preserve all analyses.
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
    }
  };
}

char Hello3::ID = 0;
static RegisterPass<Hello3>
Z("hello3", "Hello World Pass (Get direct calls)");


namespace {
  // Hello2 - The second implementation with getAnalysisUsage implemented.
  struct Hello4 : public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    Hello4() : ModulePass(ID) {}

    bool runOnModule(Module& M)override{
        setupHooks("_Z10__initMaini",M);
        Module::FunctionListType&functions=M.getFunctionList();
        for(Module::FunctionListType::iterator FI=functions.begin(),FE=functions.end();FI!=FE;++FI){
            if(FI->getName()=="_Z10__initMaini"){
                continue;
            }
            if(FI->getName()=="main"){
                instrumentEnterFunction("_Z10__initMaini",*FI,M);
            }
        }
        return true;
    }

    void setupHooks(StringRef instrumentFunctionName, Module& M){
        auto& context=M.getContext();
        Type* voidTy=Type::getVoidTy(context);
        Type* intTy=Type::getInt32Ty(context);
        FunctionType* funcTy=FunctionType::get(voidTy,intTy,false);
        Function::Create(funcTy,llvm::GlobalValue::ExternalLinkage)->setName(instrumentFunctionName);
    }

    void instrumentEnterFunction(StringRef instrumentFunctionName,Function& functionToInstrument,Module& M){
        auto& context=M.getContext();
        Type* voidTy=Type::getVoidTy(context);
        Type* intTy=Type::getInt32Ty(context);
        std::vector<Type*> params;
        params.push_back(intTy);
        FunctionType* funcTy=FunctionType::get(voidTy,params,false);
        Constant* hook=M.getOrInsertFunction(instrumentFunctionName,funcTy);
        BasicBlock* bb=&functionToInstrument.front();
        Instruction *I =&bb->front();
        std::vector<Value*> args;
        for(unsigned i=0;i<funcTy->getNumParams();++i){
            Type* t=funcTy->getParamType(i);
            Value* newValue=ConstantInt::get(t,0x3f3f3f3f);
            args.push_back(newValue);
            errs()<<"getNumParams()"<<i<<"\n";
        }
        CallInst::Create(hook,args)->insertBefore(I);
    }
  };
}

char Hello4::ID = 0;
static RegisterPass<Hello4>
W("hello4","test hello4 (set up a hook)");
