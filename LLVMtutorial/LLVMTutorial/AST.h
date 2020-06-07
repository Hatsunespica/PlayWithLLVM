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
#include "llvm/IR/Instructions.h"

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
    bool isOperator;
    unsigned precedence;
  public:
    PrototypeAST(const std::string& name,std::vector<std::string> args,bool isOperator=false,unsigned precedence=0)
        :name(name),args(args),isOperator(isOperator),precedence(precedence){};
    const std::string getName()const{return name;}
    Function* codegen();
    bool isUnaryOp()const{return isOperator&&args.size()==1;}
    bool isBinaryOp()const {return isOperator&&args.size()==2;}

    char getOperatorName()const{
        assert(isUnaryOp()||isBinaryOp());
        return name[name.size()-1];
    }

    unsigned getBinaryPrecedence()const{return precedence;}
  };

  class FunctionAST{
    std::unique_ptr<PrototypeAST> proto;
    std::unique_ptr<ExprAST> body;
  public:
    FunctionAST(std::unique_ptr<PrototypeAST> proto,
                std::unique_ptr<ExprAST> body):proto(std::move(proto)),body(std::move(body)){};
    Function* codegen();
  };

  class IfExprAST:public ExprAST{
    std::unique_ptr<ExprAST> cond,then,els;
    public:
      IfExprAST(std::unique_ptr<ExprAST> cond
        ,std::unique_ptr<ExprAST> then
        , std::unique_ptr<ExprAST>els)
        :cond(std::move(cond)),then(std::move(then)),els(std::move(els)){};
      virtual Value* codegen();
  };

  class ForExprAST:public ExprAST{
    std::string varName;
    std::unique_ptr<ExprAST> start,end,step,body;

    public:
      ForExprAST(const std::string& varName,
                std::unique_ptr<ExprAST> start,
                std::unique_ptr<ExprAST> end,
                std::unique_ptr<ExprAST> step,
                std::unique_ptr<ExprAST> body):varName(varName),start(std::move(start)),
                    end(std::move(end)),step(std::move(step)),body(std::move(body)){};
      virtual Value* codegen();
  };

  class UnaryExprAST:public ExprAST{
    char opCode;
    std::unique_ptr<ExprAST> operand;

    public:
        UnaryExprAST(char opCode,std::unique_ptr<ExprAST> operand):opCode(opCode),operand(std::move(operand)){};
        Value* codegen() override;
  };
};

extern LLVMContext theContext;
static IRBuilder<> builder(theContext);
extern std::unique_ptr<Module> theModule;
static std::map<std::string,Value*> namedValues;
extern std::unique_ptr<legacy::FunctionPassManager> theFPM;
extern std::unique_ptr<KaleidoscopeJIT> theJIT;
extern std::map<std::string, std::unique_ptr<tutorial::PrototypeAST>> functionProtos;
extern std::map<char,int> binopPrecedence;


std::unique_ptr<tutorial::ExprAST> logError(const char* str);
std::unique_ptr<tutorial::PrototypeAST> logErrorP(const char* str);
Value* logErrorV(const char* str);

#endif // AST_H_INCLUDED
