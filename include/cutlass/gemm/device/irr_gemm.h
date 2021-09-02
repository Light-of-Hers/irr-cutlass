#pragma once

#include <cmath>
#include <vector>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/numeric_types.h"

namespace cutlass {
namespace gemm {
namespace device {

template <typename EA, typename LA, typename EB, typename LB, typename EC, typename LC>
struct IrrGemmBase {
  virtual Status run(GemmCoord problem_size, TensorRef<EA const, LA> ref_A,
                     TensorRef<EB const, LB> ref_B, TensorRef<EC const, LC> ref_C,
                     TensorRef<EC, LC> ref_D) = 0;
  virtual double estimate_time(GemmCoord problem_size) = 0;
  virtual std::vector<int> get_config() = 0;
};

template <typename EA, typename LA, typename EB, typename LB, typename EC, typename LC, int BM,
          int BN, int BK, int WM, int WN, int GM, int GN>
struct IrrGemm : public IrrGemmBase<EA, LA, EB, LB, EC, LC> {
  using Base = IrrGemmBase<EA, LA, EB, LB, EC, LC>;

  using UnderlyingOp =
      Gemm<EA, LA, EB, LB, EC, LC, EC, cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
           GemmShape<BM, BN, BK>, GemmShape<WM, WN, BK>, GroupShape<GM, GN>>;

  using EpilogueOp = typename UnderlyingOp::EpilogueOutputOp;
  using Arguments = typename UnderlyingOp::Arguments;
  static int const kSharedSize = sizeof(typename UnderlyingOp::GemmKernel::Mma::SharedStorage);
  static int const kRegSize =
      sizeof(typename UnderlyingOp::GemmKernel::Mma::FragmentC) +
      sizeof(typename UnderlyingOp::GemmKernel::Mma::FragmentA) +
      sizeof(typename UnderlyingOp::GemmKernel::Mma::FragmentB) +
      sizeof(typename UnderlyingOp::GemmKernel::Mma::Operator::FragmentA) * 2 +
      sizeof(typename UnderlyingOp::GemmKernel::Mma::Operator::FragmentB) * 2;

  Status run(GemmCoord problem_size, TensorRef<EA const, LA> ref_A, TensorRef<EB const, LB> ref_B,
             TensorRef<EC const, LC> ref_C, TensorRef<EC, LC> ref_D) override {
    Arguments args{problem_size, ref_A, ref_B, ref_C, ref_D, typename EpilogueOp::Params{}, 1};
    UnderlyingOp gemm_op;
    return gemm_op(args);
  }
  double estimate_time(GemmCoord problem_size) override {
    // for V100
    constexpr double freq = 1.75;
    constexpr double mem_bandwidth = 750;
    constexpr double issue_cycle = 4;
    constexpr double mem_ld = 398;
    constexpr double departure_del_coal = 4;
    constexpr double max_shared_per_sm = 96 * (1 << 10);
    constexpr double max_regs_per_sm = 64 * (1 << 10);
    // constexpr double max_regs_per_thread = 255;
    constexpr double max_threads_per_sm = 2048;
    constexpr double n_sms = 80;

    double M = problem_size.m(), N = problem_size.n(), K = problem_size.k();
    if (std::is_same<LC, cutlass::layout::ColumnMajor>::value) {
      std::swap(M, N);
    }
    double PM = std::ceil(M / BM) * BM, PN = std::ceil(N / BN) * BN, PK = std::ceil(K / BK) * BK;
    double TM = WM / GM, TN = WN / GN;

    double n_threads_per_block = (BM / TM) * (BN / TN);
    double n_blocks = (PM / BM) * (PN / BN);

    double shared_per_block = kSharedSize;
    double n_regs_per_thread = kRegSize / 4 + 20;
    double n_active_blocks_per_sm = std::min({
        std::floor(max_threads_per_sm / n_threads_per_block),
        std::floor(std::floor(max_regs_per_sm / n_regs_per_thread) / n_threads_per_block),
        std::floor(max_shared_per_sm / shared_per_block),
    });
    double n_active_warps_per_sm = std::floor(n_active_blocks_per_sm * n_threads_per_block / 32);
    double n_active_sm = std::min(n_blocks, n_sms);

    double n_iters = PK / BK;
    double n_comp_insts_per_iter = BM * BK / n_threads_per_block + BK * BN / n_threads_per_block +
                                   TM * BK + BK * TN + TM * TN * BK + 10;
    double n_coal_mem_insts_per_iter =
        BM * BK / n_threads_per_block + BK * BN / n_threads_per_block;

    double n_comp_insts = n_comp_insts_per_iter * n_iters + (BM * BN) / n_threads_per_block;
    double n_coal_mem_insts = n_coal_mem_insts_per_iter * n_iters + (BM * BN) / n_threads_per_block;
    // double n_uncoal_mem_insts = 0;
    double n_sync_insts = n_iters;

    // double n_coal_per_mw = 32;
    // double n_uncoal_per_mw = 1;

    double load_bytes_per_warp = 4 * 32;
    double departure_delay = departure_del_coal;
    double mem_l = mem_ld;
    double mwp_without_bw_full = mem_l / departure_delay;
    double bw_per_warp = freq * load_bytes_per_warp / mem_l;
    double mwp_peak_bw = mem_bandwidth / (bw_per_warp * n_active_sm);
    double mwp = std::min({mwp_without_bw_full, mwp_peak_bw, n_active_warps_per_sm});
    double comp_cycles = issue_cycle * (n_comp_insts + n_coal_mem_insts);
    double mem_cycles = mem_l * n_coal_mem_insts;
    double cwp_full = (mem_cycles + comp_cycles) / comp_cycles;
    double cwp = std::min({cwp_full, n_active_warps_per_sm});
    double n_rep = n_blocks / (n_active_blocks_per_sm * n_active_sm);

    double exec_cycles_app = 0;
    if (mwp == n_active_warps_per_sm && cwp == n_active_warps_per_sm) {
      exec_cycles_app =
          (mem_cycles + comp_cycles + comp_cycles / n_coal_mem_insts * (mwp - 1)) * n_rep;
    } else if (cwp >= mwp || comp_cycles > mem_cycles) {
      exec_cycles_app =
          (mem_cycles * n_active_warps_per_sm / mwp + comp_cycles / n_coal_mem_insts * (mwp - 1)) *
          n_rep;
    } else {
      exec_cycles_app = (mem_l + comp_cycles * n_active_warps_per_sm) * n_rep;
    }
    double sync_cost =
        (departure_delay * (mwp - 1) * n_sync_insts * n_active_blocks_per_sm * n_rep);
    double exec_cycles_with_sync = exec_cycles_app + sync_cost;

    return exec_cycles_with_sync / freq / 6 / 1000000;
  }
  std::vector<int> get_config() override { return {BM, BN, BK, WM, WN, GM, GN}; }
};

template <typename EA, typename LA, typename EB, typename LB, typename EC, typename LC>
struct IrrGemmCollection {
  using BaseGemm = IrrGemmBase<EA, LA, EB, LB, EC, LC>;

  IrrGemmCollection() {
    gemm_ops_ = {
#define IRR_STRATEGY(BM, BN, BK, WM, WN, GM, GN)                     \
  new IrrGemm<EA, LA, EB, LB, EC, LC, BM, BN, BK, WM, WN, GM, GN>{}, \
      new IrrGemm<EA, LA, EB, LB, EC, LC, BN, BM, BK, WN, WM, GN, GM>{},
#include "cutlass/x_macro/irr_strategy.def"
#undef IRR_STRATEGY
    };
  }
  ~IrrGemmCollection() {
    for (auto p : gemm_ops_) delete p;
  }
  const std::vector<BaseGemm*>& gemm_ops() { return gemm_ops_; }

 private:
  std::vector<BaseGemm*> gemm_ops_;
};

}  // namespace device
}  // namespace gemm
}  // namespace cutlass
