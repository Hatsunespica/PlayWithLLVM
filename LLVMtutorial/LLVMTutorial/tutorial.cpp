#include "parser.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>

using namespace llvm;

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
