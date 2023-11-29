#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "mutatorUtil.h"
#include "mlir/InitAllDialects.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/ADT/StringSet.h"
#include <filesystem>
#include <string>
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "llvm/ADT/APInt.h"
#include "KnownBits.h"

using namespace std;
using namespace mlir;

llvm::cl::OptionCategory MlirTvCategory("mlir-tv options", "");
llvm::cl::OptionCategory MLIR_MUTATE_CAT("mlir-mutate-tv options", "");

llvm::cl::opt<string> filename_src(llvm::cl::Positional,
                                   llvm::cl::desc("first-mlir-file"),
                                   llvm::cl::Required,
                                   llvm::cl::value_desc("filename"),
                                   llvm::cl::cat(MLIR_MUTATE_CAT));
llvm::cl::opt<bool>
    arg_verbose("verbose", llvm::cl::desc("Be verbose about what's going on"),
                llvm::cl::Hidden, llvm::cl::init(false),
                llvm::cl::cat(MLIR_MUTATE_CAT));

llvm::cl::opt<long long> randomSeed(
    "s",
    llvm::cl::value_desc("specify the seed of the random number generator"),
    llvm::cl::cat(MLIR_MUTATE_CAT),
    llvm::cl::desc("specify the seed of the random number generator"),
    llvm::cl::init(-1));

filesystem::path inputPath, outputPath;

bool isValidInputPath();

std::string toString(const KnownBits& kb){
    string res;
    res.resize(kb.getBitWidth());
    for(size_t i=0;i<res.size();++i){
        unsigned N = res.size() - i - 1;
        if(kb.Zero[N]&&kb.One[N]){
            res[i]='!';
        }else if(kb.Zero[N]){
            res[i]='0';
        }else if(kb.One[N]){
            res[i]='1';
        }else{
            res[i]='?';
        }
    }
    return res;
}

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::EnableDebugBuffering = true;  

  llvm::cl::ParseCommandLineOptions(argc, argv);

  MLIRContext context;
  DialectRegistry registry;
  mlir::registerAllDialects(context);
  context.appendDialectRegistry(registry);
  context.allowUnregisteredDialects();

  if (!isValidInputPath()) {
    llvm::errs() << "Invalid input file!\n";
    return 1;
  }

  string errorMessage;
  auto src_file = openInputFile(filename_src, &errorMessage);

  if (!src_file) {
    llvm::errs() << errorMessage << "\n";
    return 66;
  }
  if (randomSeed >= 0) {
    util::Random::setSeed((unsigned)randomSeed);
  }
  llvm::SourceMgr src_sourceMgr;
  ParserConfig parserConfig(&context);
  src_sourceMgr.AddNewSourceBuffer(move(src_file), llvm::SMLoc());
  auto ir_before = parseSourceFile<ModuleOp>(src_sourceMgr, parserConfig);
  //ir_before->print(llvm::errs());
  ModuleOp moduleOp=ir_before.release();
  for(auto& op:moduleOp.getBodyRegion().front().getOperations()){
      if(auto funcOp=llvm::dyn_cast<func::FuncOp>(op);!funcOp.isDeclaration()){
          DataFlowSolver solver;
          dataflow::IntegerRangeAnalysis* analysis=solver.load<dataflow::IntegerRangeAnalysis>();
          //KnownBitsRangeAnalysis* knownBitsRangeAnalysis=solver.load<KnownBitsRangeAnalysis>();
          //dataflow::SparseConstantPropagation* analysis=solver.load<dataflow::SparseConstantPropagation>();
          LogicalResult result=solver.initializeAndRun(funcOp);

          //auto tmp=solver.lookupState<dataflow::Executable>(ProgramPoint(&funcOp.getBody().front()));
          //((dataflow::Executable*)tmp)->setToLive();
          funcOp->walk([&](Operation* op){
              auto tmp=solver.lookupState<dataflow::Executable>(ProgramPoint(op->getBlock()));
              if(tmp){
                  ((dataflow::Executable*)tmp)->setToLive();
              }
          });
          funcOp->walk([&](Operation* op){
             //auto res=knownBitsRangeAnalysis->visit(ProgramPoint(op));
             //if(res.failed()){
             //    llvm::errs()<<"Error\n";
             //}
             auto res=analysis->visit(ProgramPoint(op));
              if(res.failed()){
                  llvm::errs()<<"Error\n";
              }
          });
          if(result.succeeded()){
              /*
              llvm::errs()<<"Success "<<solver.analysisStates.size()<<"\n";
              auto opit=funcOp->getBlock()->getOperations().begin();
              for(auto it=solver.analysisStates.begin();it!=solver.analysisStates.end();++it){
                  it->first.first.print(llvm::errs());
                  llvm::errs()<<"\n";
                  //opit->print(llvm::errs());
                  //llvm::errs()<<"\n";
                  llvm::errs()<<"\n";
                  it->second->print(llvm::errs());
                  llvm::errs()<<"\n";
                  llvm::errs()<<(it->first.second==TypeID::get<dataflow::IntegerValueRangeLattice>());
                  llvm::errs()<<"\n";
              }*/
              llvm::errs()<<"Check args\n";
              for(auto it=funcOp.args_begin();it!=funcOp.args_end();++it){
                  ProgramPoint point(*it);
                  point.print(llvm::errs());
                  llvm::errs()<<"\n";
                  auto res=solver.lookupState<dataflow::IntegerValueRangeLattice>(point);
                  //auto res=solver.lookupState<KnownBitsRangeLattice>(point);
                  if(res!=nullptr){
                      llvm::errs()<<"Known Bits: ";
                      res->print(llvm::errs());
                      llvm::errs()<<"\n";
                  }else {
                      llvm::errs() << res << "\n";
                  }

              }
              funcOp->walk([&](Operation* op){
                  if(op->getNumResults()==0){
                      return;
                  }
                  ProgramPoint point(op->getResult(0));
                 auto intRes=solver.lookupState<dataflow::IntegerValueRangeLattice>(point);
                 auto res=solver.lookupState<KnownBitsRangeLattice>(point);
                 point.print(llvm::errs());
                 llvm::errs()<<"\n";
                 if(res!=nullptr){
                     std::string resStr=toString(res->getValue().getValue());
                     Twine tmpTwine(resStr);
                     StringAttr kb=StringAttr::get(&context, tmpTwine);
                     op->setAttr("kb", kb);
                     llvm::errs()<<"Known Bits: ";
                     res->print(llvm::errs());
                     llvm::errs()<<"\n";
                 }else{
                     llvm::errs()<<res<<"\n";
                 }
                 intRes->dump();
                 llvm::errs()<<"\n";
                 //auto it = solver.analysisStates.find({ProgramPoint(op), TypeID::get<dataflow::IntegerValueRangeLattice>()});

              });
          }else{
              llvm::errs()<<"obtaining result failed\n";
          }
      }
  }
  moduleOp.print(llvm::errs());
  return 0;
}

bool isValidInputPath() {
  bool result = filesystem::status(string(filename_src)).type() ==
                filesystem::file_type::regular;
  if (result) {
    inputPath = filesystem::path(string(filename_src));
  }
  return result;
}

