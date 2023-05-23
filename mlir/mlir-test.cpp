#include "mlir/IR/Dialect.h"
#include "mlir/Parser/Parser.h"
#include "mutatorUtil.h"
#include "mlir/InitAllDialects.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/StringSet.h"
#include <filesystem>
#include <string>

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

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::EnableDebugBuffering = true;  

  llvm::cl::ParseCommandLineOptions(argc, argv);

  MLIRContext context;
  DialectRegistry registry;
  mlir::registerAllDialects(context);
  context.appendDialectRegistry(registry);
  //context.allowUnregisteredDialects();

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
    ir_before->print(llvm::errs());

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

