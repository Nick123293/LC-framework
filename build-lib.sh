nvcc -shared -O3 -arch=sm_80 -fmad=false -DUSE_GPU \
  -Xcompiler="-O3 -march=native -fopenmp -mno-fma -ffp-contract=off -fPIC" \
  -I. \
  -o lc.so \
  lc-gpu-wrapper-compatible.cu