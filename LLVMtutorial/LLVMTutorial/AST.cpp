#include "AST.h"
#define IfNullThenReturn(ptr) if(!ptr)return nullptr;

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
            break;
    }
    Function* f=getFunction(std::string("binary")+op);
    assert(f&&"binary operator not found");
    Value* ops[2]{L,R};
    return builder.CreateCall(f,ops,"binop");
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

    if(p.isBinaryOp())
        binopPrecedence[p.getOperatorName()]=p.getBinaryPrecedence();

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

Value* IfExprAST::codegen(){
    Value* condV=cond->codegen();
    if(!condV)
        return nullptr;

    condV=builder.CreateFCmpONE(condV,ConstantFP::get(theContext,APFloat(0.0)),"ifcond");
    Function* theFunction=builder.GetInsertBlock()->getParent();

    BasicBlock* thenBB=BasicBlock::Create(theContext,"then",theFunction);
    BasicBlock* elseBB=BasicBlock::Create(theContext,"else");
    BasicBlock* mergeBB=BasicBlock::Create(theContext,"ifcont");

    builder.CreateCondBr(condV,thenBB,elseBB);
    builder.SetInsertPoint(thenBB);
    Value* thenV=then->codegen();
    if(!thenV)
        return nullptr;
    builder.CreateBr(mergeBB);

    thenBB=builder.GetInsertBlock();
    theFunction->getBasicBlockList().push_back(elseBB);
    builder.SetInsertPoint(elseBB);
    Value* elseV=els->codegen();
    if(!elseV)
        return nullptr;

    builder.CreateBr(mergeBB);
    elseBB=builder.GetInsertBlock();

    theFunction->getBasicBlockList().push_back(mergeBB);
    builder.SetInsertPoint(mergeBB);
    PHINode* pn=builder.CreatePHI(Type::getDoubleTy(theContext),2,"iftmp");
    pn->addIncoming(thenV,thenBB);
    pn->addIncoming(elseV,elseBB);
    return pn;


}

Value* ForExprAST::codegen(){
    Value* startVal=start->codegen();
    if(!startVal)
        return nullptr;

    Function* f=builder.GetInsertBlock()->getParent();
    BasicBlock* preheaderBB=builder.GetInsertBlock();
    BasicBlock* loopBB=BasicBlock::Create(theContext,"loop",f);

    builder.CreateBr(loopBB);
    builder.SetInsertPoint(loopBB);

    PHINode* var=builder.CreatePHI(Type::getDoubleTy(theContext),
                                    2,varName);
    var->addIncoming(startVal,preheaderBB);

    Value* oldVal=namedValues[varName];
    namedValues[varName]=var;

    if(!body->codegen())
        return nullptr;

    Value* stepVal=nullptr;
    if(step){
        stepVal=step->codegen();
        if(!stepVal)
            return nullptr;
    }else
        stepVal=ConstantFP::get(theContext,APFloat(1.0));

    Value* nextVar=builder.CreateFAdd(var,stepVal,"nextvar");
    Value* endCond=end->codegen();
    if(!endCond)
        return nullptr;

    endCond=builder.CreateFCmpONE(endCond,ConstantFP::get(theContext,APFloat(0.0)),"loopcond");

    BasicBlock* loopEndBB=builder.GetInsertBlock();
    BasicBlock* afterBB=BasicBlock::Create(theContext,"afterloop",f);
    builder.CreateCondBr(endCond,loopBB,afterBB);
    builder.SetInsertPoint(afterBB);
    var->addIncoming(nextVar,loopEndBB);

    if(oldVal)
        namedValues[varName]=oldVal;
    else
        namedValues.erase(varName);

    return Constant::getNullValue(Type::getDoubleTy(theContext));
}

Value* UnaryExprAST::codegen(){
    Value* operandV=operand->codegen();

    IfNullThenReturn(operandV);

    Function* f=getFunction(std::string("unary")+opCode);
    if(!f)
        return logErrorV("Unknown unary operator");
    return builder.CreateCall(f,operandV,"unop");
}
