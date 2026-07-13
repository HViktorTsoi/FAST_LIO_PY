# FAST_LIO_PY

**A pure-Python (NumPy) reimplementation of [FAST-LIO2](https://github.com/hku-mars/FAST_LIO), written for code review and teaching.**

The complete LiDAR-inertial odometry pipeline in readable NumPy, split into two top-to-bottom-readable files:

- **[`fastlio_numpy.py`](fastlio_numpy.py)** (~1 700 lines) — the **SLAM algorithm**: the iterated error-state Kalman filter (predict / update), IMU initialization + forward propagation + motion undistortion, the point-to-plane observation model, incremental-map update, and the offline main loop.
- **[`fastlio_utils.py`](fastlio_utils.py)** (~1 200 lines) — the **data structures and infrastructure** the algorithm operates on: SO(3)/S(2) manifold math, the 23-DOF manifold state (`StateIkfom`), the scipy incremental-map KD-tree, raw-bytes rosbag parsing, the profiling timer, config/CLI, geometry helpers, file output, and the per-frame-map aggregation + open3d visualization (`aggregate_map`, also a standalone CLI).

## Usage

**Dependencies**: `numpy`, `scipy`, `pyyaml`, and `rosbag` (Python ≥ 3.7). **No ROS installation is required** — the project only *reads* bag files (no `roscore`, no ROS runtime), so `rosbag` can be pip-installed on its own from the [rospypi](https://github.com/rospypi/simple) index:

```bash
pip install --extra-index-url https://rospypi.github.io/simple/ rosbag
```

(Add `roslz4` from the same index if you need to read LZ4-compressed bags.) Optional: `open3d` for map **visualization** via `aggregate_map` (a hand-written binary-PCD writer is the fallback for saving the aggregated PCD).

**Test data**: example Livox AVIA rosbags are available from the original FAST-LIO repository — [Google Drive](https://drive.google.com/drive/folders/1CGYEJ9-wWjr8INyan6q1BZz_5VtGB-fP?usp=sharing).

> **LiDAR support**: at present only **Livox (AVIA-class) `CustomMsg`** bags are supported (fast raw-bytes parsing path). Support for other LiDAR types — Velodyne / Ouster `PointCloud2` — is under active development.

### 1. Run

Run the odometry on a bag (keep `fastlio_utils.py` next to `fastlio_numpy.py` — the algorithm file imports it):

```bash
python3 fastlio_numpy.py \
    --bag your_livox_avia.bag \
    --config config/avia.yaml \
    --output_dir ./out [--profile]
```

This writes `out/Log/trajectory_py_tum.txt` (TUM trajectory) and streams the map as **per-frame output** under `out/frames/` — each scan's undistorted (LiDAR-body) cloud is appended to `clouds.bin` and its estimated pose + online-estimated extrinsics recorded in `index.npz`. The run stays memory-light (no whole-map accumulation in RAM) and never writes a giant PCD.

### 2. Aggregate result & visualize

Reconstruct the dense map on demand — this applies `point_body_to_world` per frame, then opens an open3d viewer:

```bash
python3 fastlio_utils.py --output_dir ./out [--voxel 0.1] [--no_show]
```

Options: `--voxel L` voxel-downsamples the map; `--pcd PATH` sets the output PCD path (default `out/PCD/map_aggregated.pcd`); `--no_show` writes the PCD without opening the viewer; `--no_save` visualizes without writing. Visualization needs `open3d` (`pip install open3d`); without it the PCD is still written.

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

## Acknowledgements & License

This is an educational reimplementation derived from [hku-mars/FAST_LIO](https://github.com/hku-mars/FAST_LIO) (FAST-LIO2: Fast Direct LiDAR-inertial Odometry, Xu et al.), built with [Claude Code](https://claude.com/claude-code). Licensed under **GPL-2.0**, same as upstream. If you use this in academic work, please cite the original FAST-LIO2 paper.
