# FAST_LIO_PY

**A single-file, pure-Python (NumPy) reimplementation of [FAST-LIO2](https://github.com/hku-mars/FAST_LIO), written for code review and teaching.**

The entire LiDAR-inertial odometry pipeline — manifold math, iterated error-state Kalman filter, IMU forward propagation and motion undistortion, incremental map, point-to-plane observation model, rosbag parsing, and the main loop — lives in one top-to-bottom-readable file: [`fastlio_numpy.py`](fastlio_numpy.py) (~2 700 lines). No compiled extensions, no JIT: the only algorithmic non-NumPy component is `scipy.spatial.cKDTree` for nearest-neighbor search.

## File organization (mirrors the C++ architecture)

| Section | Content | C++ counterpart |
|---|---|---|
| §1 | SO(3) / S(2) manifold math | `common_lib.h` / IKFoM (MTK) toolkit |
| §2 | 23-DOF manifold state, ⊞ / ⊟ | `include/use-ikfom.hpp` |
| §3 | Iterated error-state KF (IESKF) | `include/IKFoM_toolkit/esekfom/esekfom.hpp` |
| §4 | IMU init, forward propagation, undistortion | `include/IMU_Processing.hpp` |
| §5 | Incremental map KD-tree (scipy, lazy rebuild) | `include/ikd-Tree/` |
| §6 | Point-to-plane residuals & 12-D Jacobian | `h_share_model` in `src/laserMapping.cpp` |
| §7 | Raw-bytes rosbag message parsing | `src/preprocess.cpp` + rosbag I/O |
| §8 | Offline main loop, CLI, TUM/PCD output | `src/laserMappingOffline.cpp` |

Section banners and walkthrough commentary are written in (academic) Chinese; inline code comments are kept in English, verbatim from the development codebase.

## Correctness

The extraction was verified against the multi-module development codebase at three levels:

1. **Unit equivalence** — 5 122 comparisons across all math/filter/map/parsing functions, bit-identical (`np.array_equal`, max |diff| = 0.0).
2. **Full-pipeline** — on 6 Livox AVIA benchmark bags, the output trajectory is **byte-identical** to the multi-module numpy path.
3. Accuracy vs. the original C++ FAST-LIO2 (ATE trans. RMSE, SE(3)-aligned, same bags):

| Bag (duration) | ATE vs C++ | Wall time |
|---|---|---|
| quick-shack (49 s) | 5.87 cm | 9.9 s |
| outdoor_run_100Hz (64 s) | 4.30 cm | 48.6 s |
| outdoor_MB_10hz (141 s) | 3.87 cm | 45.6 s |
| outdoor_MB_100Hz (117 s) | 9.41 cm | 84.9 s |
| HKU_MB (260 s) | 5.17 cm | 90.0 s |
| 100hz_2021 (351 s) | 26.82 cm | 295.4 s |

Aggregate wall time is ≈ 4.8× the C++ original (574 s vs 119 s) — single-threaded BLAS is forced on purpose (thread-pool spin-up dominates compute on the small 23×23 / 12×12 / 3×3 matrices of this filter; serial BLAS measured **−60 % wall** vs default threading). The two bags above the 6.4 cm threshold sit in numerically chaotic regimes where any floating-point implementation change moves the trajectory (the C++/Python gap there is FP-accumulation noise, not an algorithmic difference).

## Usage

```bash
python3 fastlio_numpy.py \
    --bag your_livox_avia.bag \
    --config config/avia.yaml \
    --output_dir ./out [--profile]
```

Outputs: `out/Log/trajectory_py_tum.txt` (TUM format) and `out/PCD/map_offline_py.pcd` (needs `open3d`).

**Dependencies**: `numpy`, `scipy`, `pyyaml`, and ROS1 `rosbag` (file I/O only; e.g. ROS Noetic, Python ≥ 3.7). Optional: `open3d` for PCD export. Supported input: Livox `CustomMsg` bags (fast raw-bytes path) and `PointCloud2` bags (rospy fallback).

## 中文简介

本仓库将 FAST-LIO2 的完整 SLAM 管线以纯 Python (NumPy) 重写并收录于单一文件，章节组织与 C++ 版模块边界一一对应，节头附学术中文讲解，面向代码审读与教学。实现与多模块开发版在 6 个数据包上输出轨迹逐字节一致；相对 C++ 原版的轨迹精度与耗时见上表。

## Acknowledgements & License

This is an educational reimplementation derived from [hku-mars/FAST_LIO](https://github.com/hku-mars/FAST_LIO) (FAST-LIO2: Fast Direct LiDAR-inertial Odometry, Xu et al.). Licensed under **GPL-2.0**, same as upstream. If you use this in academic work, please cite the original FAST-LIO2 paper.
