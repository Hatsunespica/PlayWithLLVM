#include "AST.h"

using namespace tutorial;
using namespace llvm;

LLVMContext theContext;
std::unique_ptr<Module> theModule;
std::unique_ptr<KaleidoscopeJIT> theJIT;
std::unique_ptr<legacy::FunctionPassManager> theFPM;
std::map<std::string, std::unique_ptr<tutorial::PrototypeAST>> functionProtos;

std::unique_ptr<ExprAST> logError(const char* str){
  fprintf(stderr,"Error: %s\n",str);
  return nullptr;
}

std::unique_ptr<tutorial::PrototypeAST> logErrorP(const char* str){
 logError(str);
 return nullptr;
}

Value* logErrorV(const char* str){
    logError(str);
    return nullptr;
}

Function* getFunction(std::string name){
    if(auto* f=theModule->getFunction(name))
        return f;

    auto fi=functionProtos.find(name);
    if(fi!=functionProtos.end()){
        return fi->second->codegen();
    }
    return nullptr;
}

Value* NumberExprAST::codegen(){
    return ConstantFP::get(theContext,APFloat(val));
}

Value* VariableExprAST::codegen(){
    Value* v=namedValues[name];
    if(!v)
        logErrorV("Unknown variable name");
    return v;
}

Value* BinaryExprAST::codegen(){
    Value* L=lhs->codegen();
    Value* R=rhs->codegen();
    if(!L||!R)
        return nullptr;
    switch(op){
        case '+':
        return builder.CreateFAdd(L,R,"addtmp");
        case '-':
        return builder.CreateFSub(L,R,"subtmp");
        case '*':
        return builder.CreateFMul(L,R,"multmp");
        case '<':
        L=builder.CreateFCmpULT(L,R,"cmptmp");
        return builder.CreateUIToFP(L,Type::getDoubleTy(theContext),"booltmp");
        default:
        return logErrorV("invalid binary operator");
    }
}

Value* CallExprAST::codegen(){
    Function* calleeF=getFunction(callee);
    if(!calleeF)
        return logErrorV("Unknown function referenced");
    if(calleeF->arg_size()!=args.size())
        return logErrorV("Incorrect # arguments passed");

    std::vector<Value*> argsV;
    for(unsigned i=0,e=args.size();i!=e;++i){
        argsV.push_back(args[i]->codegen());
        if(!argsV.back())
            return nullptr;
    }
    return builder.CreateCall(calleeF,argsV,"calltmp");
}

Function* PrototypeAST::codegen(){
    std::vector<Type*> doubles(args.size(),Type::getDoubleTy(theContext));
    FunctionType* ft=FunctionType::get(Type::getDoubleTy(theContext),doubles,false);
    Function* f=Function::Create(ft,Function::ExternalLinkage,name,theModule.get());

    unsigned idx=0;
    for(auto&  arg: f->args()){
        arg.setName(args[idx++]);
    }
    return f;
}

Function* FunctionAST::codegen(){
    auto& p=*proto;
    functionProtos[proto->getName()]=std::move(proto);
    Function* theFunction=getFunction(p.getName());
    if(!theFunction)
        return nullptr;

    BasicBlock* bb=BasicBlock::Create(theContext,"entry",theFunction);
    builder.SetInsertPoint(bb);

    namedValues.clear();
    for(auto& arg: theFunction->args())
        namedValues[std::string(arg.getName())]=&arg;

    if(Value* retVal=body->codegen()){
        builder.CreateRet(retVal);
        verifyFunction(*theFunction);
        theFPM->run(*theFunction);
        return theFunction;
    }
    theFunction->eraseFromParent();
    return nullptr;
}
