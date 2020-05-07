#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>
#include <string>
#include <memory>

using namespace llvm;

enum Token{
           tok_eof=-1,
           tok_def=-2,
           tok_extern=-3,
           tok_identifier=-4,
           tok_number=-5,
};

static std::string identifierStr;
static double numVal;

static int gettok(){
  static int lastChar=' ';

  while(isspace(lastChar))lastChar=getchar();

  if(isalpha(lastChar)){
    identifierStr=lastChar;
    while(isalnum(lastChar=getchar())){
      identifierStr+=lastChar;
    }

    if(identifierStr=="def")
      return tok_def;
    else if(identifierStr=="extern")
      return tok_extern;
    return tok_identifier;
  }else if(isdigit(lastChar)||lastChar=='.'){
    std::string numStr;
    do{
      numStr+=lastChar;
      lastChar=getchar();
    }while(isdigit(lastChar)||lastChar=='.');

    numVal=strtod(numStr.c_str(),nullptr);
    return tok_number;
  }else if(lastChar=='#'){
    do{
      lastChar=getchar();
    }while(lastChar!=EOF&&lastChar!='\n'&&lastChar!='\r');
    if(lastChar!=EOF)
      return gettok();
  }

  if(lastChar==EOF)
    return tok_eof;

  int thisChar=lastChar;
  lastChar=getchar();
  return thisChar;
}

namespace{
  class ExprAST{
  public:
    virtual ~ExprAST() = default;
    virtual Value* codegen()=0;
  };

  class NumberExprAST:public ExprAST{
    double val;
  public:
    NumberExprAST(double _val):val(_val){};
    Value* codegen() override;
  };

  class VariableExprAST:public ExprAST{
    std::string name;
  public:
    VariableExprAST(const std::string& _name):name(_name){};
    Value* codegen() override;
  };

  class BinaryExprAST:public ExprAST{
    char op;
    std::unique_ptr<ExprAST> lhs,rhs;
  public:
    BinaryExprAST(char _op,std::unique_ptr<ExprAST> _lhs,
                     std::unique_ptr<ExprAST> _rhs)
      :op(_op),lhs(std::move(_lhs)),rhs(std::move(_rhs)){};
    Value* codegen() override;
  };

  class CallExprAST:public ExprAST{
    std::string callee;
    std::vector<std::unique_ptr<ExprAST>> args;
  public:
    CallExprAST(const std::string& _callee,std::vector<std::unique_ptr<ExprAST>> args):callee(_callee),args(std::move(args)){};
    Value* codegen() override;
  };

  class PrototypeAST{
    std::string name;
    std::vector<std::string> args;
  public:
    PrototypeAST(const std::string& name,std::vector<std::string> args):name(name),args(args){};
    const std::string getName()const{return name;}
    Function* codegen();
  };

  class FunctionAST{
    std::unique_ptr<PrototypeAST> proto;
    std::unique_ptr<ExprAST> body;
  public:
    FunctionAST(std::unique_ptr<PrototypeAST> proto,
                std::unique_ptr<ExprAST> body):proto(std::move(proto)),body(std::move(body)){};
    Function* codegen();
  };
}

static int curTok;
static int getNextToken(){return curTok=gettok();}

static std::map<char,int> binopPrecedence;

static int getTokPrecedence(){
  if(!isascii(curTok))
    return -1;

  int tokPrec=binopPrecedence[curTok];
  if(tokPrec<=0)
    return -1;
  return tokPrec;
}

static LLVMContext theContext;
static IRBuilder<> builder(theContext);
static std::unique_ptr<Module> theModule;
static std::map<std::string,Value*> namedValues;

std::unique_ptr<ExprAST> logError(const char* str){
  fprintf(stderr,"Error: %s\n",str);
  return nullptr;
}

std::unique_ptr<PrototypeAST> logErrorP(const char* str){
 logError(str);
 return nullptr;
}

Value*  logErrorV(const char* str){
    logError(str);
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
    Function* calleeF=theModule->getFunction(callee);
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
    Function* theFunction=theModule->getFunction(proto->getName());
    if(!theFunction)
        theFunction=proto->codegen();
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
        return theFunction;
    }
    theFunction->eraseFromParent();
    return nullptr;
}

static std::unique_ptr<ExprAST> parseExpression();

static std::unique_ptr<ExprAST> parseNumberExpr(){
  auto result=std::make_unique<NumberExprAST>(numVal);
  getNextToken();
  return std::move(result);

}


static std::unique_ptr<ExprAST> parseParenExpr(){
  getNextToken();
  auto v=parseExpression();
  if(!v)
    return nullptr;

  if(curTok!=')')
    return logError("Expected ')'");
  getNextToken();
  return v;
}

static std::unique_ptr<ExprAST> parseIdentifierExpr(){
  std::string idName=identifierStr;
  getNextToken();

  if(curTok!='(')
    return std::make_unique<VariableExprAST>(idName);

  getNextToken();
  std::vector<std::unique_ptr<ExprAST>> args;
  if(curTok!=')'){
    while(true){
      if(auto arg=parseExpression()){
        args.push_back(std::move(arg));
      }else{
        return nullptr;
      }

      if(curTok==')')
        break;
      if(curTok!=',')
        return logError("Expected ')' or ',' in argument list");
      getNextToken();
    }
  }

  getNextToken();
  return std::make_unique<CallExprAST>(idName,std::move(args));
}

static std::unique_ptr<ExprAST> parsePrimary(){
  switch(curTok){
  default:
    return logError("unknown token when expecting an expression");
  case tok_identifier:
    return parseIdentifierExpr();
  case tok_number:
    return parseNumberExpr();
  case '(':
    return parseParenExpr();
  }
}


static std::unique_ptr<ExprAST> parseBinopRHS(const int exprPrec,
                                              std::unique_ptr<ExprAST> lhs){
  while(true){
    int tokPrec=getTokPrecedence();
    if(tokPrec<exprPrec)
      return lhs;

    int binOp=curTok;
    getNextToken();

    auto rhs=parsePrimary();
    if(!rhs)
      return nullptr;

    int nextPrec=getTokPrecedence();
    if(tokPrec<nextPrec){
      rhs=parseBinopRHS(tokPrec+1,std::move(rhs));
      if(!rhs)
        return nullptr;
    }
    lhs=std::make_unique<BinaryExprAST>(binOp,std::move(lhs),
                                        std::move(rhs));
  }
}

static std::unique_ptr<ExprAST> parseExpression(){
  auto lhs=parsePrimary();
  if(!lhs)
    return nullptr;
  return parseBinopRHS(0,std::move(lhs));
}

static std::unique_ptr<PrototypeAST> parsePrototype(){
  if(curTok!=tok_identifier)
    return logErrorP("Expected function name in prototype");

  std::string fName=identifierStr;
  getNextToken();
  if(curTok!='(')
    return logErrorP("Expected '(' in prototype");

  std::vector<std::string> argNames;
  while(getNextToken()==tok_identifier)
    argNames.push_back(identifierStr);
  if(curTok!=')')
    return logErrorP("Expected ')' in prototype");

  getNextToken();
  return std::make_unique<PrototypeAST>(fName,std::move(argNames));
}


static std::unique_ptr<FunctionAST> parseDefinition(){
  getNextToken();
  auto proto=parsePrototype();
  if(!proto)
    return nullptr;

  if(auto E=parseExpression())
    return std::make_unique<FunctionAST>(std::move(proto),std::move(E));
  return nullptr;
}

static std::unique_ptr<FunctionAST> parseTopLevelExpr(){
  if(auto E=parseExpression()){
    auto proto=std::make_unique<PrototypeAST>("__anon_expr",
                                              std::vector<std::string>());
    return std::make_unique<FunctionAST>(std::move(proto),std::move(E));
  }
  return nullptr;
}

static std::unique_ptr<PrototypeAST> parseExtern(){
  getNextToken();
  return parsePrototype();
}

static void handleDefinition(){
  if(auto fnAST=parseDefinition()){
    if(auto* fnIR=fnAST->codegen()){
        fprintf(stderr, "Read a function definition. \n");
        fnIR->print(errs());
        fprintf(stderr,"\n");
    }
  }else{
    getNextToken();
  }
}

static void handleExtern(){
  if(auto protoAST=parseExtern()){
    if(auto* fnIR=protoAST->codegen()){
        fprintf(stderr,"Read an extern: \n");
        fnIR->print(errs());
        fprintf(stderr,"\n");
    }
  }else{
    getNextToken();
  }
}

static void handleTopLevelExpression(){
  if(auto fnAST=parseTopLevelExpr()){
    if(auto* fnIR=fnAST->codegen()){
        fprintf(stderr,"Read a top-level expr\n");
        fnIR->print(errs());
        fprintf(stderr,"\n");
    }
  }else{
    getNextToken();
  }
}


static void mainLoop(){
  while(true){
    fprintf(stderr,"ready> ");
    switch(curTok){
    case tok_eof:
      return;
    case ';':
      getNextToken();
      break;
    case tok_def:
      handleDefinition();
      break;
    case tok_extern:
      handleExtern();
      break;
    default:
      handleTopLevelExpression();
      break;
    }
  }
}

int main(){
  binopPrecedence['<']=10;
  binopPrecedence['+']=20;
  binopPrecedence['-']=20;
  binopPrecedence['*']=40;

  fprintf(stderr,"ready> ");
  getNextToken();
  theModule=std::make_unique<Module>("my cool jit",theContext);
  mainLoop();
  theModule->print(errs(),nullptr);
  return 0;
}
