
clang -cc1 -triple spir copyData.cl -O0 -finclude-default-header -emit-llvm-bc -o file.bc
llvm-spirv file.bc -o file.spv
echo "Generating file.spv ....... [OK]"


clang -cc1 -triple spir kernels.cl -O2 -finclude-default-header -emit-llvm-bc -o kernels.bc
llvm-spirv kernels.bc -o kernels.spv
echo "Generating kernels.spv .... [OK]"

rm file.bc kernels.bc
