import os
import sys
from argparse import ArgumentParser

DIR = os.path.dirname(__file__)
BIN_DIR = os.path.join(DIR, "./.configurable_irr_gemm_bins")
SRC_FILE = os.path.join(DIR, "./configurable_irr_gemm.cc")


def build_bin(path, E, LA, LB, LC, strategy):
    cmd_tks = [
        "nvcc -arch=sm_70 -std=c++14 -x cu -O3 -I '{}'".format(
            os.path.join(DIR, "../include")),
        SRC_FILE,
        "-o", path,
        f"-DELEMENT='{E}'",
        f"-DLAYOUT_A='{LA}'",
        f"-DLAYOUT_B='{LB}'",
        f"-DLAYOUT_C='{LC}'",
        "-DSTRATEGY='{}'".format(r'\,'.join(str(x) for x in strategy)),
    ]
    cmd = ' '.join(str(tk) for tk in cmd_tks)
    print(cmd, file=sys.stderr, flush=True)
    os.system(cmd)


def get_bin_path(dtype, layouts, strategy):
    if not os.path.exists(BIN_DIR):
        os.mkdir(BIN_DIR)
    path = os.path.join(
        BIN_DIR, "irr-gemm-{}-{}{}{}-{}".format(dtype, *(int(l == "Col") for l in layouts), '-'.join(str(x) for x in strategy)))
    if not os.path.exists(path):
        build_bin(path, dtype, *layouts, strategy)
    return path


def exec_bin_and_read_output(path, *args):
    cmd = f"{path} {' '.join(str(x) for x in args)}"
    print(cmd, file=sys.stderr, flush=True)
    with os.popen(cmd, "r") as fp:
        return fp.readlines()[0].strip()


def time_irr_gemm(dtype, layouts, strategy, shape):
    path = get_bin_path(dtype, layouts, strategy)
    return float(exec_bin_and_read_output(path, *shape, 10))


STRATEGIES = [
    (2, 64, 32, 2, 32, 2, 16),
    (2, 128, 32, 2, 64, 2, 16),
    (4, 128, 16, 4, 64, 4, 8),
    (4, 256, 16, 4, 128, 4, 8),
    (4, 128, 32, 4, 32, 4, 8),
    (8, 128, 8, 8, 64, 4, 8),
    (8, 256, 8, 8, 128, 4, 8),
    (8, 128, 16, 8, 32, 4, 8),
    (8, 256, 16, 8, 64, 4, 8),
    (16, 128, 8, 16, 32, 4, 8),
    (16, 256, 8, 16, 64, 4, 8),
    (16, 256, 16, 16, 32, 4, 8),
    (32, 128, 8, 32, 64, 4, 8),
    (32, 128, 8, 32, 32, 4, 8),
    (32, 128, 8, 16, 32, 4, 8),
    (32, 256, 8, 32, 64, 4, 8),
    (32, 256, 8, 32, 32, 4, 8),
    (64, 64, 8, 32, 64, 4, 8),
    (64, 128, 8, 32, 64, 4, 8),
    (64, 128, 8, 32, 32, 4, 8),
    (64, 256, 8, 32, 64, 4, 8),
    (128, 128, 8, 32, 64, 4, 8),
]

STRATEGIES += [
    (BN, BM, BK, WN, WM, GN, GM)
    for (BM, BN, BK, WM, WN, GM, GN) in STRATEGIES
]

SHAPES = [
    # some shapes
    (71, 56962, 70),
]


def main():
    dtype = "float"
    layouts = ("Row", "Row", "Row")
    for strategy in STRATEGIES:
        for shape in SHAPES:
            print(strategy, shape, time_irr_gemm(
                dtype, layouts, strategy, shape))


if __name__ == "__main__":
    main()
