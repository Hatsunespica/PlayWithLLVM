cmake_minimum_required(VERSION 3.13.0)
project(mlir-mutate VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)

string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

# CMake usually does decent job in finding out the proper dependency packages,
# so for most of the times you won't have to care about the <NAME>_ROOT variables.
# If you are not satisfied with the CMake's choices, you may use these variables
# to manually override them with the MLIR, Z3, or cvc5 you prefer.
set(MLIR_ROOT CACHE PATH "MLIR installation top-level directory")
option(USE_LIBC "Use libc++ in case the MLIR (and cvc5) is linked against libc++")

find_package(MLIR REQUIRED)
# MLIR_VERSION does not exist, so use LLVM_VERSION instead
# If package MLIR is found, package LLVM must have been found in the process
message(STATUS "Found MLIR ${LLVM_VERSION} from ${MLIR_DIR}/MLIRConfig.cmake")
include(AddLLVM)

# /============================================================/
# 1. Build object files to check warnings/errors before linking
# /============================================================/

set(PROJECT_OBJ "mlir-mutate")
add_library(${PROJECT_OBJ} OBJECT
    mutatorUtil.cpp
        KnownBits.cpp)

# Check MLIR and LLVM headers and include
target_include_directories(${PROJECT_OBJ} PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(${PROJECT_OBJ} PUBLIC ${MLIR_INCLUDE_DIRS})

# Warn about unused variables
target_compile_options(${PROJECT_OBJ} PUBLIC -Wunused-variable)
# Using cl::opt requires this
target_compile_options(${PROJECT_OBJ} PUBLIC -fno-rtti)

# Try using libc if possible
if(USE_LIBC)
    target_compile_options(${PROJECT_OBJ} PUBLIC -stdlib=libc++)
endif()

# /============================================================/
# 2. Build libmlirtv
# /============================================================/

set(PROJECT_LIB "mlirtv")
add_library(${PROJECT_LIB})
target_link_libraries(${PROJECT_LIB} PUBLIC ${PROJECT_OBJ})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIB_LIST ${dialect_libs})
target_link_libraries(${PROJECT_LIB} PUBLIC ${LIB_LIST} pthread m curses)
llvm_update_compile_flags(${PROJECT_LIB})

# Try using libc if possible
if(USE_LIBC)
    target_link_options(${PROJECT_LIB} PUBLIC -stdlib=libc++)
endif()

# /============================================================/
# 3. Build executable
# /============================================================/

# Build executable
add_executable(mlir-test mlir-test.cpp)
target_link_libraries(mlir-test ${PROJECT_LIB})
add_executable(mlir-random mlir-random.cpp)
target_link_libraries(mlir-random ${PROJECT_LIB})

