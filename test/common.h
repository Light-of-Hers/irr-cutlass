#ifndef __MINI_BENCH_CUTLASS_COMMON_H__
#define __MINI_BENCH_CUTLASS_COMMON_H__

#include <cassert>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <iostream>
#include <unistd.h>
#include <vector>

#include "./utils.h"

#define CUDA_CHECK(err)                                                        \
  RZ_CHECK(err == cudaSuccess, "%s", cudaGetErrorString(err))

#define CUTLASS_CHECK(stat)                                                    \
  RZ_CHECK(stat == cutlass::Status::kSuccess, "%s",                            \
           cutlass::cutlassGetStatusString(stat))

double time_cutlass_gemm(int M, int K, int N, int number = 10);

#endif