#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/irr_gemm.h>
#include <cutlass/tensor_view.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/gemm.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_copy.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/host/tensor_foreach.h>
#include <cutlass/util/tensor_view_io.h>

#include <algorithm>
#include <cassert>
#include <cmath>

#include "common.h"

using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;

#ifndef ELEMENT
#define ELEMENT float
#endif
#ifndef LAYOUT_A
#define LAYOUT_A Row
#endif
#ifndef LAYOUT_B
#define LAYOUT_B Row
#endif
#ifndef LAYOUT_C
#define LAYOUT_C Row
#endif
#ifndef STRATEGY
#define STRATEGY 128, 128, 8, 32, 64, 4, 8
#endif

using Gemm = cutlass::gemm::device::IrrGemm<ELEMENT, LAYOUT_A, ELEMENT, LAYOUT_B, ELEMENT, LAYOUT_C,
                                            STRATEGY>;

double test_irr_gemm(int M, int N, int K, int number) {
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::HostTensor<ELEMENT, LAYOUT_A> tensor_a(problem_size.mk());
  cutlass::HostTensor<ELEMENT, LAYOUT_B> tensor_b(problem_size.kn());
  cutlass::HostTensor<ELEMENT, LAYOUT_C> tensor_c(problem_size.mn());
  cutlass::HostTensor<ELEMENT, LAYOUT_C> tensor_d(problem_size.mn());
  cutlass::HostTensor<ELEMENT, LAYOUT_C> tensor_ref_d(problem_size.mn());

  cutlass::reference::host::TensorFillRandomUniform(tensor_a.host_view(), 1, 10, -10, 0);
  cutlass::reference::host::TensorFillRandomUniform(tensor_b.host_view(), 1, 10, -10, 0);
  cutlass::reference::host::TensorFillRandomUniform(tensor_c.host_view(), 1, 10, -10, 0);
  cutlass::reference::host::TensorFill(tensor_d.host_view());
  cutlass::reference::host::TensorFill(tensor_ref_d.host_view());

  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  auto dev_a = tensor_a.device_ref();
  auto dev_b = tensor_b.device_ref();
  auto dev_c = tensor_c.device_ref();
  auto dev_d = tensor_d.device_ref();

  cutlass::Status stat;
  cutlass::reference::device::Gemm<ELEMENT, LAYOUT_A, ELEMENT, LAYOUT_B, ELEMENT, LAYOUT_C, ELEMENT,
                                   ELEMENT>
      gemm_device;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  DEFER(cudaEventDestroy(start), cudaEventDestroy(stop));

  Gemm gemm_op;
  stat = gemm_op.run(problem_size, dev_a, dev_b, dev_c, dev_d), CUTLASS_CHECK(stat);

  ELEMENT alpha = 1, beta = 0;
  // Launch device reference gemm kernel
  gemm_device(problem_size, alpha, tensor_a.device_ref(), tensor_b.device_ref(), beta,
              tensor_c.device_ref(), tensor_ref_d.device_ref());
  // Wait for kernels to finish
  cudaDeviceSynchronize();
  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d.sync_host();
  tensor_ref_d.sync_host();
  bool passed =
      cutlass::reference::host::TensorEquals(tensor_d.host_view(), tensor_ref_d.host_view());
  RZ_CHECK(passed, "wrong answer");

  double total_ms = 0;
  for (int i = 0; i < number; ++i) {
    cudaEventRecord(start);
    stat = gemm_op.run(problem_size, dev_a, dev_b, dev_c, dev_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CUTLASS_CHECK(stat);
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    total_ms += ms;
  }
  return total_ms / number;
}

int main(int argc, char** argv) {
  assert(argc > 3);
  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  int number = 10;
  if (argc > 4) number = atoi(argv[4]);
  double ms = test_irr_gemm(M, N, K, number);
  printf("%lf\n", ms);
}
