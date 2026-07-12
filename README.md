# FAST_LIO_PY

**A pure-Python (NumPy) reimplementation of [FAST-LIO2](https://github.com/hku-mars/FAST_LIO), written for code review and teaching.**

The complete LiDAR-inertial odometry pipeline in readable NumPy, split into two top-to-bottom-readable files:

- **[`fastlio_numpy.py`](fastlio_numpy.py)** (~1 700 lines) — the **SLAM algorithm**: the iterated error-state Kalman filter (predict / update), IMU initialization + forward propagation + motion undistortion, the point-to-plane observation model, incremental-map update, and the offline main loop.
- **[`fastlio_utils.py`](fastlio_utils.py)** (~1 200 lines) — the **data structures and infrastructure** the algorithm operates on: SO(3)/S(2) manifold math, the 23-DOF manifold state (`StateIkfom`), the scipy incremental-map KD-tree, raw-bytes rosbag parsing, the profiling timer, config/CLI, geometry helpers, file output, and the per-frame-map aggregation + open3d visualization (`aggregate_map`, also a standalone CLI).

The split follows a simple principle — *`utils` holds the objects; `numpy` holds the algorithm that acts on them.* No compiled extensions, no JIT: the only algorithmic non-NumPy component is `scipy.spatial.cKDTree` for nearest-neighbor search. Section banners and walkthrough commentary are in academic Chinese; inline code comments are English, verbatim from the development codebase.

## File organization (mirrors the C++ architecture)

| Where | Content | C++ counterpart |
|---|---|---|
| `utils` §A | SO(3) / S(2) manifold math | `common_lib.h` / IKFoM (MTK) toolkit |
| `utils` §B | 23-DOF manifold state, ⊞ / ⊟ | `include/use-ikfom.hpp` |
| `numpy` §3 | Iterated error-state KF (IESKF) | `include/IKFoM_toolkit/esekfom/esekfom.hpp` |
| `numpy` §4 | IMU init, forward propagation, undistortion | `include/IMU_Processing.hpp` |
| `utils` | Incremental map KD-tree (scipy, lazy rebuild) | `include/ikd-Tree/` |
| `numpy` §6 | Point-to-plane residuals & 12-D Jacobian | `h_share_model` in `src/laserMapping.cpp` |
| `utils` §rosbag | Raw-bytes rosbag message parsing | `src/preprocess.cpp` + rosbag I/O |
| `numpy` §8 | Offline main loop, CLI, TUM/PCD output | `src/laserMappingOffline.cpp` |

## Accuracy & timing vs the original C++ FAST-LIO2

Same-machine comparison on 6 Livox AVIA bags (ATE = translational RMSE, SE(3)-aligned; wall = internal SLAM loop):

| Bag (duration) | rate | ATE vs C++ | C++ | NumPy | **real-time** |
|---|---|---|---|---|---|
| quick-shack (49 s) | 10 Hz | 5.87 cm | 1.6 s | 9.7 s | **5.0×** |
| outdoor_MB_10hz (141 s) | 10 Hz | 3.87 cm | 12.3 s | 43.5 s | **3.2×** |
| HKU_MB (260 s) | 10 Hz | 5.17 cm | 23.9 s | 88.7 s | **2.9×** |
| outdoor_run_100Hz (64 s) | 100 Hz | 4.30 cm | 7.2 s | 47.2 s | 1.4× |
| outdoor_MB_100Hz (117 s) | 100 Hz | 9.41 cm | 11.6 s | 81.0 s | 1.4× |
| 100hz_2021 (351 s) | 100 Hz | 26.82 cm | 42.1 s | 285.3 s | 1.2× |

**Real-time:** even in this most-conservative configuration (single-threaded BLAS, no JIT), the pure-NumPy pipeline runs **faster than real time on every bag**. At 10 Hz it has 3–5× headroom (~20–34 ms/scan vs the 100 ms budget); at 100 Hz it is still real-time but tight (~7–8 ms/scan vs 10 ms). (These are offline timings; a live ROS node adds per-message deserialization overhead, small relative to the SLAM compute.)

**Timing:** aggregate ≈ 5–6× the C++ wall. Single-threaded BLAS is forced on purpose — thread-pool spin-up dominates compute on this filter's small 23×23 / 12×12 / 3×3 matrices; serial BLAS measured **−60 % wall** vs default threading. The remaining gap is the per-scan Python-dispatch floor over the sequential, small-matrix filter, not algorithmic overhead.

**Accuracy:** four of six bags are within a few cm of C++. The two outliers sit in numerically chaotic regimes — `outdoor_MB_100Hz` is a global gravity-init tilt (trajectory *shape* is fine), `100hz_2021` is long-run drift — where any floating-point implementation change moves the trajectory; the C++/Python gap there is FP-accumulation noise, not an algorithmic difference.

## Correctness

The extraction was verified against the multi-module development codebase at three levels:

1. **Unit equivalence** — 5 122 comparisons across all math / filter / map / parsing functions, bit-identical (`np.array_equal`, max |diff| = 0.0).
2. **Full-pipeline** — on the 6 benchmark bags, the output trajectory is **byte-identical** to the multi-module NumPy path.
3. **Accuracy vs C++** — the table above.

## Usage

**Run** the odometry on a bag (keep `fastlio_utils.py` next to `fastlio_numpy.py` — the algorithm file imports it):

```bash
python3 fastlio_numpy.py \
    --bag your_livox_avia.bag \
    --config config/avia.yaml \
    --output_dir ./out [--profile]
```

This writes `out/Log/trajectory_py_tum.txt` (TUM trajectory) and streams the map as **per-frame output** under `out/frames/` — each scan's undistorted (LiDAR-body) cloud is appended to `clouds.bin` and its estimated pose + online-estimated extrinsics recorded in `index.npz`. The run stays memory-light (no whole-map accumulation in RAM) and never writes a giant PCD.

**Aggregate & visualize** the dense map on demand — this reconstructs the world-frame map by applying `point_body_to_world` per frame, then opens an open3d viewer:

```bash
python3 fastlio_utils.py --output_dir ./out [--voxel 0.1] [--no_show]
```

Options: `--voxel L` voxel-downsamples the map; `--pcd PATH` sets the output PCD path (default `out/PCD/map_aggregated.pcd`); `--no_show` writes the PCD without opening the viewer; `--no_save` visualizes without writing. Visualization needs `open3d` (`pip install open3d`); without it the PCD is still written.

**Dependencies**: `numpy`, `scipy`, `pyyaml`, and ROS1 `rosbag` (file I/O only; e.g. ROS Noetic, Python ≥ 3.7). Optional: `open3d` for map **visualization** via `aggregate_map` (a hand-written binary-PCD writer is the fallback for saving the aggregated PCD). Supported input: Livox `CustomMsg` bags (fast raw-bytes path) and `PointCloud2` bags (rospy fallback).

## 中文简介

本仓库将 FAST-LIO2 的完整 SLAM 管线以纯 Python (NumPy) 重写，面向代码审读与教学，拆为两个自顶向下可读的文件：`fastlio_numpy.py`（SLAM 算法本体：IESKF、IMU 处理、点面观测、主循环）与 `fastlio_utils.py`（其操作的数据结构与基础设施：SO(3)/S(2) 数学、流形状态、增量地图 KD-Tree、rosbag 解析、计时、配置、IO）。章节组织与 C++ 版模块边界对应，节头附学术中文讲解。地图采用**按帧流式输出**（每帧去畸变点云 + 位姿写入 `out/frames/`，运行内存轻量）；跑完用 `python3 fastlio_utils.py --output_dir <dir>` 聚合重建稠密地图并经 open3d 可视化。实现与多模块开发版在 6 个数据包上输出轨迹逐字节一致；纯 NumPy 在 10 Hz 下有 3–5× 实时余量。精度与耗时见上表。

## Acknowledgements & License

This is an educational reimplementation derived from [hku-mars/FAST_LIO](https://github.com/hku-mars/FAST_LIO) (FAST-LIO2: Fast Direct LiDAR-inertial Odometry, Xu et al.). Licensed under **GPL-2.0**, same as upstream. If you use this in academic work, please cite the original FAST-LIO2 paper.
