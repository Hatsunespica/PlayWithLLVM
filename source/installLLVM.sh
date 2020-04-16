svn co https://user@llvm.org/svn/llvm-project/llvm/tags/RELEASE_500/final llvm
cd llvm/tools
svn co http://llvm.org/svn/llvm-project/cfe/tags/RELEASE_500/final clang
cd clang/tools # (To be clear, you are now in llvm/tools/clang/tools)
svn co http://llvm.org/svn/llvm-project/clang-tools-extra/tags/RELEASE_500/final extra
cd ../../../../llvm/projects # (To be clear, you are now in llvm/projects
svn co http://llvm.org/svn/llvm-project/compiler-rt/tags/RELEASE_500/final compiler-rt
cd ../../.. #(You are now in your desktop directory)
mkdir build #(if you have not already done so)
cd build #(You are now in your build directory)
cmake -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_TARGET_ARCH=X86 -DCMAKE_BUILD_TYPE="Release" -DLLVM_BUILD_EXAMPLES=1 -DCLANG_BUILD_EXAMPLES=1 -G "Unix Makefiles" ../source/llvm/
'make -j 8' #(from within the build directory to start the process)
