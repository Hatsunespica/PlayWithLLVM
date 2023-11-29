#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
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
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "llvm/ADT/APInt.h"

using namespace std;
using namespace mlir;

llvm::cl::OptionCategory MlirTvCategory("mlir-tv options", "");
llvm::cl::OptionCategory MLIR_MUTATE_CAT("mlir-mutate-tv options", "");

llvm::cl::opt<string> filename_src(llvm::cl::Positional,
                                   llvm::cl::desc("first-mlir-file"),
                                   llvm::cl::Required,
                                   llvm::cl::value_desc("filename"),
                                   llvm::cl::cat(MLIR_MUTATE_CAT));

filesystem::path inputPath, outputPath;

bool isValidInputPath();

mlir::BlockArgument addParameter(mlir::func::FuncOp& func, mlir::Type ty){
    func.insertArgument(func.getNumArguments(), ty, {}, func->getLoc());
    return func.getArgument(func.getNumArguments()-1);
}

void addResult(mlir::func::FuncOp& func, mlir::Value val){
    func.insertResult(func->getNumResults(),val.getType(),{});
    mlir::Operation& retOp=func.getFunctionBody().getBlocks().front().back();
    retOp.insertOperands(retOp.getNumOperands(), val);
}

mlir::func::FuncOp moveToFunc(MLIRContext& context,llvm::SmallVector<mlir::Operation*> ops, mlir::Location loc){
    mlir::FunctionType funcTy=mlir::FunctionType::get(&context,{},{});
    auto func=mlir::func::FuncOp::create(loc,"tmp",funcTy);
    mlir::Block* blk=func.addEntryBlock();

    mlir::OpBuilder builder(&context);
    auto retOp=builder.create<mlir::func::ReturnOp>(func->getLoc());
    blk->push_back(retOp.getOperation());

    unordered_set<mlir::Operation*> needReturn;
    unordered_map<mlir::Operation*, mlir::Operation*> um;
    //arg_num -> current arg_num;
    unordered_map<int, mlir::BlockArgument> arg_um;
    std::vector<mlir::Operation*> stk;

    for(auto op:ops){
        mlir::Operation* cur=op->clone();
        stk.push_back(cur);
        um.insert({op, cur});
        needReturn.insert(cur);

        for(size_t i=0;i<op->getNumOperands();++i){
            mlir::Value arg=op->getOperand(i);
            mlir::Type arg_ty = arg.getType();
            if(mlir::Operation* definingOp=arg.getDefiningOp();definingOp){
                if(auto it=um.find(definingOp);it!=um.end()){
                    /*
                     * Calc the result index in definingOp
                     * Assume there are multiple returns
                     */
                    size_t idx=0;
                    needReturn.erase(it->second);
                    for(;idx<definingOp->getNumResults();++idx){
                        if(definingOp->getResult(idx)==arg){
                            cur->setOperand(i,it->second->getResult(idx));
                        }
                    }
                }else{
                    mlir::BlockArgument newArg= addParameter(func,arg_ty);
                    cur->setOperand(i, newArg);
                }
            }else{
                mlir::BlockArgument blk_arg=arg.cast<mlir::BlockArgument>();
                int arg_num=blk_arg.getArgNumber();
                if(arg_um.find(arg_num)==arg_um.end()){
                    arg_um.insert({arg_num,addParameter(func, arg_ty)});
                }
                cur->setOperand(i,arg_um[arg_num]);
            }
        }
    }

    while(!stk.empty()){
        blk->push_front(stk.back());
        stk.pop_back();
    }

    for(auto op:needReturn){
        for(auto res_it=op->result_begin();res_it!=op->result_end();++res_it){
            addResult(func,*res_it);
        }
    }

    return func;
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
    context.loadDialect<mlir::arith::ArithDialect>();

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

    llvm::SourceMgr src_sourceMgr;
    ParserConfig parserConfig(&context);
    src_sourceMgr.AddNewSourceBuffer(move(src_file), llvm::SMLoc());
    auto ir_before = parseSourceFile<ModuleOp>(src_sourceMgr, parserConfig);
    //ir_before->print(llvm::errs());
    ModuleOp moduleOp=ir_before.release();
    auto i32=mlir::IntegerType::get(moduleOp.getContext(), 32);
    mlir::IntegerAttr int0Attr=mlir::IntegerAttr::get(i32, 0);
    mlir::IntegerAttr int1Attr=mlir::IntegerAttr::get(i32, 1);

    mlir::OpBuilder builder(&context);
    auto const0Op=builder.create<mlir::arith::ConstantOp>(moduleOp->getLoc(),i32,int0Attr);
    auto const1Op=builder.create<mlir::arith::ConstantOp>(moduleOp->getLoc(),i32,int1Attr);
    auto addRes=builder.create<mlir::arith::AddIOp>(moduleOp->getLoc(),const0Op.getResult(), const1Op.getResult());
    auto subRes=builder.create<mlir::arith::SubIOp>(moduleOp->getLoc(), addRes.getResult(), const0Op.getResult());
    auto mulRes=builder.create<mlir::arith::MulIOp>(moduleOp->getLoc(), const1Op.getResult(), const0Op.getResult());
    llvm::SmallVector<mlir::Operation*> res{const0Op.getOperation(),const1Op.getOperation(),addRes.getOperation(),subRes.getOperation(),mulRes.getOperation()};
    auto func= moveToFunc(context, res,moduleOp->getLoc());
    func.dump();
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

