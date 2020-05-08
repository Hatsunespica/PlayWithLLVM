#ifndef AST_H_INCLUDED
#define AST_H_INCLUDED

#include "llvm/IR/Value.h"
#include <string>
#include <memory>
#include <vector>
#include "llvm/IR/LLVMContext.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "KaleidoscopeJIT.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"

using namespace llvm;
using namespace llvm::orc;

namespace tutorial{
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

extern LLVMContext theContext;
static IRBuilder<> builder(theContext);
extern std::unique_ptr<Module> theModule;
static std::map<std::string,Value*> namedValues;
extern std::unique_ptr<legacy::FunctionPassManager> theFPM;
extern std::unique_ptr<KaleidoscopeJIT> theJIT;
extern std::map<std::string, std::unique_ptr<tutorial::PrototypeAST>> functionProtos;


std::unique_ptr<tutorial::ExprAST> logError(const char* str);
std::unique_ptr<tutorial::PrototypeAST> logErrorP(const char* str);
Value* logErrorV(const char* str);

#endif // AST_H_INCLUDED
