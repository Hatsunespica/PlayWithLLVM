#include "parser.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>

using namespace llvm;

static void initModuleAndPassManager(){
    theModule=std::make_unique<Module>("my cool jit",theContext);
    theModule->setDataLayout(theJIT->getTargetMachine().createDataLayout());
    theFPM=std::make_unique<legacy::FunctionPassManager>(theModule.get());
    theFPM->add(createInstructionCombiningPass());
    theFPM->add(createReassociatePass());
    theFPM->add(createGVNPass());
    theFPM->add(createCFGSimplificationPass());
    theFPM->doInitialization();
}

static void handleDefinition(){
  if(auto fnAST=parseDefinition()){
    if(auto* fnIR=fnAST->codegen()){
        fprintf(stderr, "Read a function definition. \n");
        fnIR->print(errs());
        fprintf(stderr,"\n");
        theJIT->addModule(std::move(theModule));
        initModuleAndPassManager();
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
        functionProtos[protoAST->getName()]=std::move(protoAST);
    }
  }else{
    getNextToken();
  }
}

static void handleTopLevelExpression(){
  if(auto fnAST=parseTopLevelExpr()){
    if(auto tmp=fnAST->codegen()){
        tmp->print(errs());
        fprintf(stderr,"\n");
        auto h=theJIT->addModule(std::move(theModule));
        initModuleAndPassManager();
        auto exprSymbol=theJIT->findSymbol("__anon_expr");
        assert(exprSymbol&&"Function not found");
        double (*FP)()=(double(*)())(intptr_t)cantFail(exprSymbol.getAddress());
        fprintf(stderr,"Evaluated to %f\n",FP());
        theJIT->removeModule(h);
    }
  }else{
    getNextToken();
  }
}

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif


extern "C" DLLEXPORT double putchard(double x){
    fputc((char)x,stderr);
    return 0;
}

extern "C" DLLEXPORT double printd(double x){
    fprintf(stderr,"%f\n",x);
    return 0;
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
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  binopPrecedence['<']=10;
  binopPrecedence['+']=20;
  binopPrecedence['-']=20;
  binopPrecedence['*']=40;

  fprintf(stderr,"ready> ");
  getNextToken();

  theJIT=std::make_unique<KaleidoscopeJIT>();
  initModuleAndPassManager();
  mainLoop();
  return 0;
}
