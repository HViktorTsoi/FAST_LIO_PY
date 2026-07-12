#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAST-LIO2 纯 Python (NumPy) 单文件实现 —— 供审读与教学.

本文件将 FAST-LIO2 离线建图管线的纯 numpy 路径完整收录于单一文件, 自顶向下
阅读即可理解整个系统。所有算法代码与 scripts/fastlio/ 包中的 numpy 路径
逐字节一致 (bit-identical), 并通过两级验收:
  (1) 单元等价测试 —— 与包内实现在相同输入下输出逐位一致;
  (2) 全量测试 —— 6 个 rosbag 上输出轨迹与包路径逐字节相同。
与原包的差异仅为: 剔除 numba JIT 内核、IPC 序列化、C++ ikd-Tree 后端等
可选组件, 只保留单一 numpy 参考路径。

章节组织参照 C++ 版 FAST-LIO 的模块边界:

    本文件章节                          对应 C++ 文件
    ------------------------------------------------------------------
    §1  SO(3)/S(2) 流形数学             common_lib.h / IKFoM (MTK) 工具箱
    §2  23 维流形状态与 ⊞/⊟ 运算        include/use-ikfom.hpp
    §3  迭代误差状态卡尔曼滤波 (IESKF)   include/IKFoM_toolkit/esekfom/esekfom.hpp
    §4  IMU 初始化・前向传播・去畸变     include/IMU_Processing.hpp
    §5  增量式地图 KD-Tree (scipy 实现)  include/ikd-Tree/ (C++ 中为独立库)
    §6  点到平面观测模型 h_share_model   src/laserMapping.cpp (函数内嵌于主文件)
    §7  rosbag 裸字节解析               src/preprocess.cpp + rosbag I/O
    §8  离线主流程                      src/laserMappingOffline.cpp

依赖: numpy, scipy, pyyaml, rosbag (仅文件 I/O; 容器内 pip 依赖见
scripts/fastlio/requirements.txt)。不依赖 numba、pybind11 或任何自研编译组件。

用法:
    python3 scripts/fastlio_numpy.py \
        --bag /mnt/ipc/bags/2020-09-16-quick-shack.bag \
        --config config/avia.yaml \
        --output_dir /tmp/out [--profile]

输出: {output_dir}/Log/trajectory_py_tum.txt (TUM 格式轨迹),
      {output_dir}/PCD/map_offline_py.pcd (点云地图, 需 open3d)。
"""
from __future__ import annotations

import os

# Force single-threaded BLAS (OpenBLAS / MKL / BLIS) BEFORE importing numpy.
# OpenBLAS spawns its own thread pool for every matrix op, which on our tiny
# matrices (23×23 EKF, 12×12 Woodbury, 3×3 SO3 correction) pays far more
# thread-spinup cost than compute. Measured -60% wall on outdoor_run_100Hz
# (102.4s → 41.5s) just from forcing serial BLAS. C++ doesn't hit this
# because Eigen inlines small matrix ops without threading.
# `setdefault` lets users override from the shell if they want to experiment.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
import time

import numpy as np
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

# 工具函数下沉：SO(3)/S(2) 流形数学、rosbag 解析、计时、配置/CLI、文件输出、
# 体素降采样等与主算法正交的符号已移入同目录 fastlio_utils.py（逐字节不变的
# 重构，无循环依赖）。此处以健壮方式（脚本自身目录入 sys.path）反向导入。
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from fastlio_utils import (hat, exp, log, A_matrix, quat_to_rot, rot_to_quat, _inv3,
    s2_Bx, s2_Nx_yy, s2_Mx, G_m_s2, _S2_LENGTH, _S2_TYP, _LIVOX_POINT_DTYPE,
    _parse_header, parse_imu_bytes, parse_livox_bytes, imu_msg_to_dict, _LIVOX_GETTER,
    livox_msg_to_array, pcl2_to_array, _try_import_rosbag, PhaseTimer, _TimerCtx,
    _NullTimer, _NullCtx, _NULL_CTX, _NULL_TIMER, load_config, get_param,
    _build_arg_parser, _save_trajectory, _save_pcd, voxel_downsample,
    StateIkfom, IkdTreeBase, IkdTreeScipy, point_body_to_world, lasermap_fov_segment)

# =============================================================================
# §1  SO(3) / S(2) 流形数学基础  →  见 fastlio_utils.py §A
# -----------------------------------------------------------------------------
# （原模块 docstring：SO(3) math utilities mirroring so3_math.h from FAST-LIO.）
#
# 本节职责：
#   提供 FAST-LIO2 状态估计所依赖的两类流形运算原语——
#   (1) 特殊正交群 SO(3) 上的基本算子：反对称阵 hat(·)、指数映射 exp(·)
#       （Rodrigues 公式）、对数映射 log(·)，以及右雅可比 A_matrix(·)，
#       另含四元数与旋转矩阵的相互转换 quat_to_rot / rot_to_quat；
#   (2) 单位球面流形 S(2)（重力向量约束 ||g|| = const）的切空间工具：
#       切空间基 s2_Bx、协方差传播矩阵 s2_Nx_yy（2×3）与 s2_Mx（3×2）。
#
# 与 C++ 版的对应关系：
#   - hat / exp / log / A_matrix 对应 include/common_lib.h 及
#     IKFoM_toolkit 中的 SO(3) 工具函数（Exp/Log/A_matrix）；
#   - s2_Bx / s2_Nx_yy / s2_Mx 对应 IKFoM_toolkit/mtk/types/S2.hpp 中
#     S2::S2_Bx / S2_Nx_yy / S2_Mx 方法。
#
# 关键公式与约定：
#   - 指数映射（Rodrigues）：exp(φ) = I + sinθ·K + (1−cosθ)·K²，
#     其中 θ = ||φ||，K = hat(φ/θ)；θ < 1e-7 时退化为单位阵。
#   - 右雅可比：A(φ) = I + (1−cosθ)/θ²·hat(φ) + (θ−sinθ)/θ³·hat(φ)²。
#   - S(2) 参数化：FAST-LIO 的重力状态取 MTK::S2<double, 98090, 10000, 1>，
#     即球面半径 length = 98090/10000 = 9.809、基准轴类型 s2_typ = 1（x 轴）；
#     若误用 s2_typ = 3 会在默认重力 [0,0,-9.81] 处触及南极奇点。
#   - 四元数一律采用 [x, y, z, w] 顺序（与 scipy / Eigen 存储一致）。
#
# 实现注记：log(·) 刻意使用 math.acos / math.sin 标量运算而非 numpy /
# scipy.Rotation，以规避每次调用的 numpy 分派开销（性能关键路径）。
# rot_to_quat 与 s2_Mx 内部的 scipy 惰性 import 按原样保留。
#
# 本节符号（hat / exp / log / A_matrix / quat_to_rot / rot_to_quat /
# s2_Bx / s2_Nx_yy / s2_Mx 及常量 G_m_s2 / _S2_LENGTH / _S2_TYP）已下沉至
# fastlio_utils.py §A，并在文件头 import 区反向导入。
# =============================================================================
# =============================================================================
# §2  23 维流形状态定义与 ⊞/⊟（boxplus / boxminus）运算
# -----------------------------------------------------------------------------
# 对应 C++ 文件：include/use-ikfom.hpp（MTK_BUILD_MANIFOLD 宏定义的 state_ikfom），
# 其中 S2 流形部分对应 include/IKFoM_toolkit/mtk/types/S2.hpp。
#
# 本节职责：
#   1. 定义误差状态迭代卡尔曼滤波（IESEKF）所操作的复合流形状态 StateIkfom。
#      流形结构为 R^3 × SO(3) × SO(3) × R^3 × R^3 × R^3 × R^3 × S2，
#      误差状态（切空间）维数为 23；名义状态若以四元数存旋转则为 24 维，
#      本实现直接以 (3,3) 旋转矩阵存储 SO(3) 分量。
#   2. 实现流形上的 ⊞（boxplus）与 ⊟（boxminus）运算：
#        - 向量分量（pos/offset_T/vel/bg/ba）：普通欧氏加减；
#        - SO(3) 分量（rot/offset_R）：R ⊞ δθ = R·Exp(δθ)，
#          R1 ⊟ R2 = Log(R2ᵀ·R1)；
#        - S2 分量（重力 grav）：g ⊞ δ = Exp(Bx(g)·δ)·g，其中 Bx(g)∈R^{3×2}
#          为 g 处切平面的一组正交基（见 so3_math 节的 s2_Bx）；
#          ⊟ 为对应的测地线对数映射（含 v_sin→0 的对径点/重合点分支）。
#
# 状态布局（承自原模块 docstring，23 DOF / 24 DIM）：
#   pos         [0:3]   (3 DOF, DIM 0:3)
#   rot         [3:6]   SO3 (3 DOF, DIM 3:7 quaternion but 3 DOF)
#   offset_R    [6:9]   SO3 (3 DOF)
#   offset_T    [9:12]  (3 DOF)
#   vel         [12:15] (3 DOF)
#   bg          [15:18] (3 DOF)
#   ba          [18:21] (3 DOF)
#   grav        [21:23] S2  (2 DOF, 3 DIM)
#
# 关键常量说明：S2 球面半径 _S2_LENGTH = 98090/10000 = 9.809，与 C++ 端
#   MTK::S2<double, 98090, 10000, 1> 严格一致；_S2_TYP = 1 表示以 x 轴为
#   基准轴构造切空间基。注意：默认重力 [0, 0, -9.81] 若取 s2_typ=3 会恰好
#   落在南极奇异点，故必须取 s2_typ=1（boxplus/boxminus 往返误差 ≈ 9e-8）。
# =============================================================================

# 常量 G_m_s2 / _S2_LENGTH / _S2_TYP 及 StateIkfom（23 维流形状态与 ⊞/⊟ 运算）
# 已下沉至 fastlio_utils.py §B，见文件头 import。
# =============================================================================
# §3  迭代误差状态卡尔曼滤波 IESKF（Iterated Error-State Kalman Filter）
# =============================================================================
# 【本节职责】
#   实现流形（manifold）上的迭代扩展卡尔曼滤波，是 C++ 版 IKFoM 工具箱
#   include/IKFoM_toolkit/esekfom/esekfom.hpp 的 Python 移植；状态动力学
#   雅可比（原 use-ikfom.hpp 中的 get_f / df_dx / df_dw）已按稀疏结构
#   内联展开于 predict() 之中，不再以独立函数形式存在。
#
# 【对应 C++ 文件】
#   - esekfom.hpp: predict() / update_iterated_dyn_share_modified()
#   - use-ikfom.hpp: 连续时间动力学 f、∂f/∂x、∂f/∂w（已内联）
#
# 【关键数据结构】
#   状态 StateIkfom（流形上 23 自由度 / 切空间 24 维）：
#     pos[0:3], rot[3:6](SO3), offset_R[6:9](SO3),
#     offset_T[9:12], vel[12:15], bg[15:18], ba[18:21], grav[21:23](S2)
#   过程噪声 12 自由度：
#     ng[0:3], na[3:6], nbg[6:9], nba[9:12]
#
# 【关键公式】
#   预测步：x ← x ⊞ f(x,u)·dt；P ← F P Fᵀ + (dt·G) Q (dt·G)ᵀ，
#           其中 F = F_x1 + f_x·dt，F_x1 在 SO3 / S2 子块处带切空间修正。
#   更新步（迭代）：以 Woodbury 形式计算卡尔曼增益，
#           A = P⁻¹ + Hᵀ H / R（23×23），求解 A·[K_x | Kh] = rhs；
#           由于 H = [h_x | 0]（h_x 为 N×12 点到平面雅可比），
#           右端项仅 13 列非零，K_x 的 12:23 列恒为零；
#           dx_ = Kh + (K_x − I)·dx_new，随后 boxplus 回流形。
#
# （原模块级 docstring，照录如下）
# Iterated EKF on manifold — Python port of IKFoM esekfom.hpp.
#
# State: StateIkfom (23 DOF)
#   pos[0:3], rot[3:6](SO3), offset_R[6:9](SO3),
#   offset_T[9:12], vel[12:15], bg[15:18], ba[18:21], grav[21:23](S2)
#
# Process noise: 12 DOF
#   ng[0:3], na[3:6], nbg[6:9], nba[9:12]
# =============================================================================


# Module constants reused across predict/update calls.
_EYE_23 = np.eye(23)
_EYE_3  = np.eye(3)
_EYE_12 = np.eye(12)


# ---------------------------------------------------------------------------
# Small-matrix helpers (avoid numpy per-call overhead for 3×3 / 2×2 inverses)
# ---------------------------------------------------------------------------
# _inv3（3×3 余子式显式求逆）已下沉至 fastlio_utils.py §A，见文件头 import。


# ---------------------------------------------------------------------------
# State DOF/DIM index mappings (must match state_ikfom manifold layout)
# ---------------------------------------------------------------------------
# SO3 states: (dof_idx, dim_idx)  in the 23-DOF / 24-DIM vectors
_SO3_STATES = [(3, 3), (6, 6)]   # rot, offset_R
# S2 state:  (dof_idx, dim_idx)
_S2_STATES  = [(21, 21)]          # grav


# ---------------------------------------------------------------------------
# predict(): IMU forward propagation + covariance update
# ---------------------------------------------------------------------------

def predict(
    state: StateIkfom,
    dt: float,
    Q: np.ndarray,
    acc: np.ndarray,
    gyro: np.ndarray,
) -> StateIkfom:
    """Functional (pure) EKF prediction: propagate a COPY of `state` forward by
    dt and return it, leaving the input object untouched. Mirrors
    esekfom::predict() in esekfom.hpp.

    Thin wrapper over predict_inplace(): both share a single implementation of
    the manifold state advance + covariance propagation
    P = F @ P @ F^T + (dt·G) @ Q @ (dt·G)^T. Use this pure variant at scan-end
    and in unit tests (no shared mutable state, safe to compare against a clean
    reference); use predict_inplace() directly in the hot IMU forward loop,
    where its module-level scratch-buffer reuse removes ~15 allocations/scan.
    Bit-identical to the fully-inlined implementation.
    """
    return predict_inplace(state.copy(), dt, Q, acc, gyro)


# Cached zero 2-vec used inside predict() to avoid reallocating each call.
_ZERO_VEC2 = np.zeros(2)


# ---------------------------------------------------------------------------
# predict_inplace(): allocation-free predict for the IMU forward loop.
# A micro-benchmark showed predict() spends ~94% of its time on allocation /
# small-op construction and only ~6% on the (irreducible) 23×23 covariance
# matmuls. The forward loop calls predict ~15×/scan, so eliminating the
# per-call allocations (state.copy of the 23×23 P, np.eye, np.zeros, matmul
# temporaries) recovers ~1-2 ms/scan on the numpy path.
#
# This mutates `state` IN PLACE and reuses module-level scratch — it is NOT
# reentrant. Safe under the serial (OMP_NUM_THREADS=1) pipeline; the pure
# functional predict() above is kept for the scan-end call and unit tests.
# ---------------------------------------------------------------------------
_PI_n     = 23
_PI_F_x1  = np.eye(_PI_n)                 # identity; only [3:6,3:6],[21:23,21:23] rewritten
_PI_f_x   = np.zeros((_PI_n, _PI_n))      # constant I at [0:3,12:15]; rest rewritten per call
_PI_f_x[0:3, 12:15] = np.eye(3)
_PI_f_w   = np.zeros((_PI_n, 12))         # constant I blocks; [3:6,0:3],[12:15,3:6] per call
_PI_f_w[15:18, 6:9]  = np.eye(3)
_PI_f_w[18:21, 9:12] = np.eye(3)
_PI_F     = np.empty((_PI_n, _PI_n))
_PI_dtG   = np.empty((_PI_n, 12))
_PI_mm1   = np.empty((_PI_n, _PI_n))      # F @ P
_PI_mm2   = np.empty((_PI_n, _PI_n))      # (F @ P) @ F^T
_PI_gq    = np.empty((_PI_n, 12))         # dtG @ Q
_PI_gqg   = np.empty((_PI_n, _PI_n))      # (dtG @ Q) @ dtG^T


def predict_inplace(state: StateIkfom, dt: float, Q: np.ndarray,
                    acc: np.ndarray, gyro: np.ndarray) -> StateIkfom:
    """In-place, allocation-free equivalent of predict(). Mutates and returns
    `state`. Bit-identical to predict() (same op order); see module scratch
    note above for the non-reentrancy caveat."""
    R        = state.rot                    # OLD rotation (rebinding state.rot later keeps this valid)
    grav     = state.grav
    acc_corr = acc - state.ba
    gyr_corr = gyro - state.bg
    vel_dot  = R @ acc_corr + grav

    # New state components (computed from OLD state before any rebind).
    new_pos = state.pos + state.vel * dt
    new_rot = R @ exp(gyr_corr * dt)
    new_vel = state.vel + vel_dot * dt

    # --- F_x1 blocks ---
    seg_rot = -gyr_corr * dt
    _PI_F_x1[3:6, 3:6] = exp(seg_rot)
    Nx = s2_Nx_yy(grav, _S2_LENGTH, _S2_TYP)
    Mx = s2_Mx(grav, _ZERO_VEC2, _S2_LENGTH, _S2_TYP)
    _PI_F_x1[21:23, 21:23] = Nx @ Mx

    # --- f_x rewritten entries (constant [0:3,12:15]=I stays) ---
    A_rot = A_matrix(seg_rot)
    _PI_f_x[3:6, 15:18]   = -A_rot
    _PI_f_x[12:15, 3:6]   = -R @ hat(acc_corr)
    _PI_f_x[12:15, 18:21] = -R
    _PI_f_x[12:15, 21:23] = -hat(grav) @ s2_Bx(grav, _S2_LENGTH, _S2_TYP)

    # --- f_w rewritten entries (constant I blocks stay) ---
    _PI_f_w[3:6, 0:3]   = -A_rot
    _PI_f_w[12:15, 3:6] = -R

    # F = F_x1 + f_x*dt ; dtG = dt*f_w  (addition commutes → bit-identical).
    # Use np.add(out=) rather than `+=` so _PI_F stays a module global (an
    # augmented assignment would make it a function-local → UnboundLocalError).
    np.multiply(_PI_f_x, dt, out=_PI_F)
    np.add(_PI_F, _PI_F_x1, out=_PI_F)
    np.multiply(_PI_f_w, dt, out=_PI_dtG)

    # P = F @ P @ F^T + dtG @ Q @ dtG^T  (same left-to-right order as predict())
    np.matmul(_PI_F, state.P, out=_PI_mm1)
    np.matmul(_PI_mm1, _PI_F.T, out=_PI_mm2)
    np.matmul(_PI_dtG, Q, out=_PI_gq)
    np.matmul(_PI_gq, _PI_dtG.T, out=_PI_gqg)

    state.P   = _PI_mm2 + _PI_gqg           # fresh array (scratch reused next call)
    state.pos = new_pos
    state.rot = new_rot
    state.vel = new_vel
    return state


# ---------------------------------------------------------------------------
# update_iterated_dyn_share_modified(): iterated EKF measurement update
# ---------------------------------------------------------------------------

# 【讲解】迭代测量更新：每次迭代先计算 dx = s ⊟ x_prop（当前估计相对传播
# 状态的切空间误差），再对 P 与 P⁻¹ 同步施加 SO3/S2 切空间修正（利用
# P_c = B P B ᵀ ⇒ P_c⁻¹ = B⁻ᵀ P⁻¹ B⁻¹ 的块对角结构，避免每轮重新求逆）。
# 增益以 Woodbury 形式经 23×23 线性方程组求解——此处必须用带选主元的
# LU（np.linalg.solve），详见函数内关于 Cholesky 数值稳定性的注释。

def update_iterated(
    state: StateIkfom,
    state_propagated: StateIkfom,
    h_share_fn: Callable,
    R_scalar: float,
    max_iter: int = 4,
    limit: float = 0.001,
) -> StateIkfom:
    """Iterated EKF update step.

    Mirrors update_iterated_dyn_share_modified() from esekfom.hpp.

    Args:
        state:            current state (will be updated in-place conceptually)
        state_propagated: state after predict() (frozen reference)
        h_share_fn:       callable(state, converge) → (h_x, h, valid)
                          h_x: (N, 12) Jacobian
                          h:   (N,)   residuals
                          valid: bool
        R_scalar:         per-observation noise variance (scalar, e.g. LASER_POINT_COV)
        max_iter:         maximum iterations
        limit:            convergence threshold on |dx|

    Returns:
        Updated StateIkfom.
    """
    n = 23
    P_propagated     = state_propagated.P.copy()
    P_propagated_inv = np.linalg.inv(P_propagated)  # computed ONCE per scan
    s = state.copy()
    t_converge = 0

    # --- Scan-level work buffers, reused across all 5 EKF iters ---
    # These replaced per-iteration `.copy()`/`np.zeros` calls. Each iter does
    # `np.copyto(dst, src)` (in-place assign) or `buf.fill(0.)` at the start,
    # saving ~6 × (23×23) allocations/scan + the 23×13 rhs.
    P_work       = np.empty((n, n), dtype=np.float64)
    P_inv_work   = np.empty((n, n), dtype=np.float64)
    A_kf_buf     = np.empty((n, n), dtype=np.float64)
    rhs13_buf    = np.empty((n, 13), dtype=np.float64)
    dx_new_buf   = np.empty(n, dtype=np.float64)
    L_buf        = np.empty((n, n), dtype=np.float64)

    # --- Pure numpy reference path ---
    for i in range(-1, max_iter):
        dx = s.boxminus(state_propagated)   # 23-vec
        np.copyto(dx_new_buf, dx)
        dx_new = dx_new_buf

        converge = (t_converge > 0)
        h_x, h, valid = h_share_fn(s, converge)

        if not valid:
            continue

        # Restore P from propagated and apply SO3/S2 tangent-space corrections.
        # Simultaneously maintain P_inv = corrected P^{-1} using the
        # block-diagonal structure:  P_c = B @ P_prop @ B^T  →
        #   P_c^{-1} = B^{-T} @ P_prop^{-1} @ B^{-1}
        # where B is identity except 3×3 (SO3) and 2×2 (S2) diagonal blocks.
        np.copyto(P_work,     P_propagated)
        np.copyto(P_inv_work, P_propagated_inv)
        P     = P_work
        P_inv = P_inv_work

        # SO3 tangent-space correction
        for (idx, _) in _SO3_STATES:
            seg     = dx[idx:idx+3]
            A_T     = A_matrix(seg).T           # (3,3)
            A_T_inv = _inv3(A_T)               # explicit cofactor — no LAPACK call
            dx_new[idx:idx+3]    = A_T     @ dx[idx:idx+3]
            P[idx:idx+3, :]      = A_T     @ P[idx:idx+3, :]
            P[:, idx:idx+3]      = P[:, idx:idx+3] @ A_T.T
            # P_inv correction: apply B^{-1} (cols) then B^{-T} (rows)
            P_inv[:, idx:idx+3]  = P_inv[:, idx:idx+3] @ A_T_inv
            P_inv[idx:idx+3, :]  = A_T_inv.T @ P_inv[idx:idx+3, :]

        # S2 tangent-space correction
        for (idx, _) in _S2_STATES:
            seg2 = dx[idx:idx+2]
            Nx   = s2_Nx_yy(s.grav, _S2_LENGTH, _S2_TYP)
            Mx   = s2_Mx(state_propagated.grav, seg2, _S2_LENGTH, _S2_TYP)
            T22  = Nx @ Mx
            # 2×2 explicit inverse (avoids np.linalg.inv call)
            det  = T22[0, 0] * T22[1, 1] - T22[0, 1] * T22[1, 0]
            T22_inv = np.array([[ T22[1, 1], -T22[0, 1]],
                                 [-T22[1, 0],  T22[0, 0]]]) / det
            dx_new[idx:idx+2]    = T22     @ dx[idx:idx+2]
            P[idx:idx+2, :]      = T22     @ P[idx:idx+2, :]
            P[:, idx:idx+2]      = P[:, idx:idx+2] @ T22.T
            P_inv[:, idx:idx+2]  = P_inv[:, idx:idx+2] @ T22_inv
            P_inv[idx:idx+2, :]  = T22_inv.T @ P_inv[idx:idx+2, :]

        # Woodbury Kalman gain — work directly with h_x (N×12) to avoid
        # allocating the padded H (N×23) matrix.  H = [h_x | zeros] so
        # H^T H has non-zeros only in the leading 12×12 block and
        # H^T h has non-zeros only in the leading 12 rows. The solution
        # K_x[:, 12:n] is therefore identically zero and we solve only
        # the 13 non-zero RHS columns (12 for K_x + 1 for Kh).
        s_R = 1.0 / R_scalar
        hxTh    = h_x.T @ h                          # (12,)
        hxThx_R = h_x.T @ h_x * s_R                  # (12, 12)

        # A_kf = P^{-1} + H^T H/R  (23×23)
        np.copyto(A_kf_buf, P_inv)
        A_kf_buf[:12, :12] += hxThx_R

        # Reduced RHS: 23 × 13 instead of 23 × 24.
        rhs13_buf.fill(0.0)
        rhs13_buf[:12, :12] = hxThx_R
        rhs13_buf[:12, 12]  = hxTh * s_R

        # LU solve. A_kf is SPD in theory — earlier we tried scipy's
        # cho_factor (on a symmetrized copy) and measured only a modest
        # isolated gain on 23×23. When combined with the serialized-BLAS
        # configuration (see OMP_NUM_THREADS=1 at offline_main.py top),
        # Cholesky's accumulated bit-difference from LU produced a
        # catastrophic ATE divergence on short IMU-init-sensitive bags
        # (quick-shack 5.79 → 62 cm). LU with pivoting is more stable.
        sol13 = np.linalg.solve(A_kf_buf, rhs13_buf)
        K_x_12 = sol13[:, :12]                       # (23, 12)
        Kh     = sol13[:, 12]                        # (23,)

        # dx_ = Kh + (K_x - I) @ dx_new
        #     = Kh + K_x_12 @ dx_new[:12] - dx_new    (K_x cols 12:n are zero)
        dx_ = Kh + K_x_12 @ dx_new[:12] - dx_new     # (23,)

        # Apply boxplus correction (in-place; s is already our working copy)
        s.boxplus(dx_)

        # Check convergence
        if np.all(np.abs(dx_) <= limit):
            t_converge += 1

        if t_converge > 1 or i == max_iter - 1:
            # Final covariance update — K_x_12 was returned by cho_solve, so
            # it lives in rhs13_buf's first 12 columns. We must finish using
            # it before the next fill, but since we return immediately below
            # that's fine.
            np.copyto(L_buf, P)
            L = L_buf

            # SO3 correction on L
            for (idx, _) in _SO3_STATES:
                seg = dx_[idx:idx+3]
                A_T = A_matrix(seg).T
                L[idx:idx+3, :]     = A_T @ P[idx:idx+3, :]
                L[:, idx:idx+3]     = L[:, idx:idx+3] @ A_T.T
                P[:, idx:idx+3]     = P[:, idx:idx+3] @ A_T.T

            # S2 correction on L
            for (idx, _) in _S2_STATES:
                seg2 = dx_[idx:idx+2]
                Nx  = s2_Nx_yy(s.grav, _S2_LENGTH, _S2_TYP)
                Mx  = s2_Mx(state_propagated.grav, seg2, _S2_LENGTH, _S2_TYP)
                T22 = Nx @ Mx
                L[idx:idx+2, :]    = T22 @ P[idx:idx+2, :]
                L[:, idx:idx+2]    = L[:, idx:idx+2] @ T22.T
                P[:, idx:idx+2]    = P[:, idx:idx+2] @ T22.T

            # K_x full = [K_x_12 | zeros(23, 11)]  →  K_x @ P uses only first
            # 12 rows of P. The subtraction allocates a fresh (23,23); s.P
            # owns that fresh memory and is unaffected by next scan's
            # L_buf / rhs13_buf reuse.
            s.P = L - K_x_12 @ P[:12, :]

            return s

    return s
# =============================================================================
# §4  IMU 处理：初始化、前向传播与运动畸变校正
# =============================================================================
# 本节为 C++ 版 include/IMU_Processing.hpp（ImuProcess 类）的纯 Python 移植，
# 职责可分为三个阶段：
#
#   (1) 静态初始化 imu_init()：对前若干帧 IMU 读数做递推均值统计，
#       以 grav = -mean_acc/|mean_acc| * G 估计重力方向（S2 流形上的初值），
#       以 bg = mean_gyr 估计陀螺零偏，并按 C++ IMU_init() 逐块设置初始
#       协方差 P（rot 块 1e-5、外参块 1e-5/1e-4… 见代码）。累积帧数超过
#       MAX_INI_COUNT 后置 imu_need_init_ = False，进入正常工作模式。
#
#   (2) 前向传播（undistort_pcl() 前半）：对落在本帧扫描区间内的 IMU 序列
#       取相邻两样本中值 a = (a_i + a_{i+1})/2、w = (w_i + w_{i+1})/2，
#       逐区间调用 predict_inplace() 完成 ESEKF 名义状态与协方差的传播；
#       每个区间末端的 {偏移时间, 世界系加速度, 去偏角速度, 速度, 位置, 姿态}
#       以 SoA（struct-of-arrays）数组 pose_* 记录——语义上等价于 C++ 中的
#       Pose6D 链 IMUpose，但省去了逐对象构造。区间首端早于上帧扫描结束
#       时刻 last_lidar_end_time_ 时，dt 被钳位到 tail - last_end，与 C++
#       UndistortPcl 的处理一致；最后再向扫描结束时刻补一次 predict()。
#
#   (3) 反向逐点去畸变（undistort_pcl() 后半）：点云按 curvature（毫秒级
#       时间偏移）排序后，对每个点用 searchsorted 定位其所属 IMU 区间
#       （head/tail 位姿），以 Rodrigues 公式
#           exp(w·dt) q = q·cosθ + (k×q)·sinθ + k·(k·q)·(1-cosθ),
#           k = w/|w|, θ = |w|·dt
#       将点从采样时刻补偿到扫描结束时刻，对应 C++ 中
#           P_comp = R_L^I^T ( R_end^T ( R_i·exp(w·dt)·(R_L^I·p + t_L^I) + T_ei ) - t_L^I ),
#       其中 T_ei = pos_i + vel_i·dt + 0.5·acc_i·dt² - pos_end。
#       全过程对 M 个点一次性向量化完成（einsum 批量矩阵-向量乘）。
#
# 顶层驱动 process() 对应 ImuProcess::Process()：初始化未完成时仅累积统计
# 并返回 (None, state)；完成后进入 undistort_pcl 返回去畸变点云与新状态。
#
# （原模块 docstring 说明：本模块是 IMU_Processing.hpp 中
#   ImuProcess::UndistortPcl() 的 Python 移植。单文件版已移除 numba 加速
#   后端（FASTLIO_ACCEL 门控 / _numba_kernels.predict_chain）与 IPC 对拍
#   入口（_ipc_server_loop），仅保留纯 numpy 参考实现。）
# =============================================================================



# ---------------------------------------------------------------------------
# ImuProcess Python class
# ---------------------------------------------------------------------------
class ImuProcess:
    """Python equivalent of ImuProcess class from IMU_Processing.hpp."""

    def __init__(self):
        self.b_first_frame_ = True
        self.imu_need_init_ = True
        self.init_iter_num  = 1
        self.start_timestamp_ = -1.0
        self.last_lidar_end_time_ = 0.0
        self.mean_acc  = np.array([0.0, 0.0, -1.0])
        self.mean_gyr  = np.zeros(3)
        self.angvel_last  = np.zeros(3)
        self.acc_s_last   = np.zeros(3)
        self.first_lidar_time = 0.0
        self.last_imu_: Optional[dict] = None

        # Extrinsics (set via set_extrinsic)
        self.Lidar_T_wrt_IMU = np.zeros(3)
        self.Lidar_R_wrt_IMU = np.eye(3)

        # Covariances
        self.cov_acc      = np.array([0.1, 0.1, 0.1])
        self.cov_gyr      = np.array([0.1, 0.1, 0.1])
        self.cov_bias_gyr = np.array([1e-4, 1e-4, 1e-4])
        self.cov_bias_acc = np.array([1e-4, 1e-4, 1e-4])

        self._MAX_INI_COUNT = 10
        self.lidar_type = 1   # 1=Livox, 2=Velodyne, 3=Ouster, 4=MARSIM

    def set_extrinsic(self, T: np.ndarray, R: Optional[np.ndarray] = None):
        self.Lidar_T_wrt_IMU = np.asarray(T).ravel()
        self.Lidar_R_wrt_IMU = np.asarray(R) if R is not None else np.eye(3)

    # _make_Q：按 use-ikfom.hpp 的 process_noise_cov() 组装 12×12 过程噪声
    # 对角阵（顺序：gyr、acc、bias_gyr、bias_acc，各 3×3 对角块）。
    def _make_Q(self) -> np.ndarray:
        Q = np.zeros((12, 12))
        Q[0:3, 0:3]   = np.diag(self.cov_gyr)
        Q[3:6, 3:6]   = np.diag(self.cov_acc)
        Q[6:9, 6:9]   = np.diag(self.cov_bias_gyr)
        Q[9:12, 9:12] = np.diag(self.cov_bias_acc)
        return Q

    # ------------------------------------------------------------------
    # IMU_init: accumulate statistics and initialize state
    # ------------------------------------------------------------------
    def imu_init(self, meas: dict, state: StateIkfom) -> StateIkfom:
        """Accumulate IMU measurements for gravity/bias initialization.

        meas keys: imu_stamps(N,), imu_acc(N,3), imu_gyro(N,3)
        Returns updated state (after MAX_INI_COUNT frames).
        """
        imu_acc  = meas["imu_acc"]    # (N, 3)
        imu_gyro = meas["imu_gyro"]   # (N, 3)

        if self.b_first_frame_:
            self._reset()
            self.init_iter_num = 1
            self.b_first_frame_ = False
            self.mean_acc = imu_acc[0].copy()
            self.mean_gyr = imu_gyro[0].copy()
            self.first_lidar_time = meas.get("lidar_beg", 0.0)

        N = self.init_iter_num
        for acc_i, gyr_i in zip(imu_acc, imu_gyro):
            N += 1
            self.mean_acc += (acc_i - self.mean_acc) / N
            self.mean_gyr += (gyr_i - self.mean_gyr) / N

        self.init_iter_num = N

        if N > self._MAX_INI_COUNT:
            # Scale to actual gravity
            cov_acc_scale = (G_m_s2 / np.linalg.norm(self.mean_acc)) ** 2
            state.grav = -self.mean_acc / np.linalg.norm(self.mean_acc) * G_m_s2
            state.bg   = self.mean_gyr.copy()
            state.offset_T = self.Lidar_T_wrt_IMU.copy()
            state.offset_R = self.Lidar_R_wrt_IMU.copy()

            # Initial P (matches C++ IMU_init)
            P = np.eye(23)
            P[6:9, 6:9]   = 1e-5 * np.eye(3)
            P[9:12, 9:12] = 1e-5 * np.eye(3)
            P[15:18, 15:18] = 1e-4 * np.eye(3)
            P[18:21, 18:21] = 1e-3 * np.eye(3)
            P[21:23, 21:23] = 1e-5 * np.eye(2)
            state.P = P
            self.imu_need_init_ = False

        return state

    # ------------------------------------------------------------------
    # UndistortPcl
    # ------------------------------------------------------------------
    def undistort_pcl(
        self,
        meas: dict,
        state: StateIkfom,
    ) -> tuple[np.ndarray, StateIkfom]:
        """IMU forward propagation + per-point motion undistortion.

        meas keys:
          imu_stamps  (N,)
          imu_acc     (N, 3)   [m/s²]
          imu_gyro    (N, 3)   [rad/s]
          lidar_pts   (M, 4)   [x, y, z, curvature_ms]
          lidar_beg   scalar   [s]
          lidar_end   scalar   [s]

        state: current StateIkfom (with P)

        Returns:
          pts_out (M, 4): undistorted points
          state:          updated state after propagation to scan end
        """
        imu_stamps = np.asarray(meas["imu_stamps"])
        imu_acc    = np.asarray(meas["imu_acc"])
        imu_gyro   = np.asarray(meas["imu_gyro"])
        lidar_pts  = np.asarray(meas["lidar_pts"], dtype=np.float64)
        pcl_beg    = float(meas["lidar_beg"])
        pcl_end    = float(meas["lidar_end"])

        # Prepend last IMU measurement (head of this frame)
        if self.last_imu_ is not None:
            imu_stamps = np.concatenate([[self.last_imu_["stamp"]], imu_stamps])
            imu_acc    = np.vstack([self.last_imu_["acc"],  imu_acc])
            imu_gyro   = np.vstack([self.last_imu_["gyro"], imu_gyro])

        imu_beg_time = imu_stamps[0]
        imu_end_time = imu_stamps[-1]

        # Sort point cloud by curvature (timestamp offset in ms)
        sort_idx = np.argsort(lidar_pts[:, 3])
        pts = lidar_pts[sort_idx].copy()

        # IMU poses are kept as struct-of-arrays (SoA), filled below once the
        # number of valid intervals is known. Index 0 is the scan-begin pose
        # (current state). This replaces a per-scan list of ~20 Pose6D objects
        # + 6 np.stack() with direct array writes — faster on BOTH the numpy and
        # numba paths. The backward warp reads acc/gyr only via tail index
        # kp_idx >= 1, so pose_acc[0]/pose_gyr[0] are never used.
        pose0_acc = self.acc_s_last.copy()
        pose0_gyr = self.angvel_last.copy()
        pose0_vel = state.vel.copy()
        pose0_pos = state.pos.copy()
        pose0_rot = state.rot.copy()

        Q = self._make_Q()

        # --- Forward propagation: batch all IMU intervals ---
        # Build dts / accs / gyros arrays outside the loop. This is the
        # single hot loop of the pipeline (10-20 iterations per scan).
        mean_acc_norm = float(np.linalg.norm(self.mean_acc))
        acc_scale = (G_m_s2 / mean_acc_norm) if mean_acc_norm > 0.01 else 1.0

        K_imu = len(imu_stamps) - 1
        # Average consecutive IMU samples for each interval
        acc_avr_all  = 0.5 * (imu_acc[:-1] + imu_acc[1:])    # (K_imu, 3)
        gyro_avr_all = 0.5 * (imu_gyro[:-1] + imu_gyro[1:])  # (K_imu, 3)

        # dts: if head_t < last_lidar_end_time_, clamp dt to tail - last_end
        head_t_arr = imu_stamps[:-1]
        tail_t_arr = imu_stamps[1:]
        dts_all = tail_t_arr - head_t_arr
        clip_mask = head_t_arr < self.last_lidar_end_time_
        if clip_mask.any():
            dts_all = dts_all.copy()
            dts_all[clip_mask] = tail_t_arr[clip_mask] - self.last_lidar_end_time_

        # Skip intervals where tail_t < last_lidar_end_time_ (both endpoints
        # before scan begin).
        skip_mask = tail_t_arr < self.last_lidar_end_time_
        valid_mask = ~skip_mask

        if valid_mask.any():
            v_idx     = np.where(valid_mask)[0]
            v_dts     = dts_all[v_idx]
            v_accs    = acc_avr_all[v_idx]
            v_gyros   = gyro_avr_all[v_idx]
            v_tail_t  = tail_t_arr[v_idx]
            Kv        = v_dts.shape[0]
            N_poses   = Kv + 1
            pose_offset = np.empty(N_poses)
            pose_offset[0]  = 0.0
            pose_offset[1:] = v_tail_t - pcl_beg

            # --- Pure-numpy reference path ---
            pose_pos = np.empty((N_poses, 3));    pose_pos[0] = pose0_pos
            pose_rot = np.empty((N_poses, 3, 3)); pose_rot[0] = pose0_rot
            pose_vel = np.empty((N_poses, 3));    pose_vel[0] = pose0_vel
            pose_acc = np.empty((N_poses, 3));    pose_acc[0] = pose0_acc
            pose_gyr = np.empty((N_poses, 3));    pose_gyr[0] = pose0_gyr
            in_acc  = v_accs[-1].copy()
            in_gyro = v_gyros[-1].copy()
            for ki in range(Kv):
                acc_avr    = v_accs[ki] * acc_scale
                angvel_avr = v_gyros[ki]
                dt = v_dts[ki]
                in_acc  = acc_avr
                in_gyro = angvel_avr

                # In-place, allocation-free predict for the hot forward loop
                # (bit-identical to predict(); ~15 calls/scan). The scan-end
                # call below keeps the functional predict().
                state = predict_inplace(state, dt, Q, in_acc, in_gyro)

                # Write the per-interval pose row directly (SoA).
                pose_gyr[ki + 1] = angvel_avr - state.bg
                pose_acc[ki + 1] = state.rot @ (acc_avr - state.ba) + state.grav
                pose_vel[ki + 1] = state.vel
                pose_pos[ki + 1] = state.pos
                pose_rot[ki + 1] = state.rot
        else:
            # No valid IMU interval — only the scan-begin pose exists, so the
            # warp (N_poses < 2) early-returns. Build the 1-row SoA for that.
            pose_offset = np.zeros(1)
            pose_pos = pose0_pos[None]; pose_rot = pose0_rot[None]
            pose_vel = pose0_vel[None]; pose_acc = pose0_acc[None]
            pose_gyr = pose0_gyr[None]
            in_acc  = imu_acc[-1] * acc_scale
            in_gyro = imu_gyro[-1]

        # Propagate to exact scan end time (always runs in pure-numpy —
        # it's a single extra predict call, cheap enough)
        note = 1.0 if pcl_end > imu_end_time else -1.0
        dt = note * (pcl_end - imu_end_time)
        state = predict(state, dt, Q, in_acc, in_gyro)

        # Save last IMU for next scan
        self.last_imu_ = {
            "stamp": imu_stamps[-1],
            "acc":   imu_acc[-1].copy(),
            "gyro":  imu_gyro[-1].copy(),
        }
        self.last_lidar_end_time_ = pcl_end
        self.angvel_last = in_gyro - state.bg
        self.acc_s_last  = state.rot @ (in_acc - state.ba) + state.grav

        # -----------------------------------------------------------------
        # Backward per-point undistortion (non-MARSIM)
        # -----------------------------------------------------------------
        if self.lidar_type == 4:   # MARSIM: no undistortion
            return pts, state

        pts_out = pts.copy()
        imu_state_end = state   # state at scan-end frame

        # ------------------------------------------------------------------
        # Vectorised undistortion: process ALL M points at once instead of
        # looping over IMU intervals × points.
        # ------------------------------------------------------------------
        N_poses = len(pose_offset)
        if N_poses < 2:
            return pts_out, state

        # SoA pose arrays built during forward propagation (no Pose6D list, no
        # np.stack). imu_offsets[0]==0; pos/rot/vel index 0 = scan-begin pose.
        imu_offsets = pose_offset    # (N,)
        rot_arr = pose_rot           # (N, 3, 3)
        vel_arr = pose_vel           # (N, 3)
        pos_arr = pose_pos           # (N, 3)
        acc_arr = pose_acc           # (N, 3)
        gyr_arr = pose_gyr           # (N, 3)

        pt_offsets = pts_out[:, 3] / 1000.0  # (M,) time offset in seconds

        # For each point, find the IMU interval it belongs to:
        # interval kp = searchsorted(imu_offsets, pt_offset, 'right')
        # head = kp-1, tail = kp
        kp_idx = np.searchsorted(imu_offsets, pt_offsets, side='right')
        kp_idx = np.clip(kp_idx, 1, N_poses - 1)   # (M,)
        head_idx = kp_idx - 1                        # (M,)

        R_imu   = rot_arr[head_idx]      # (M, 3, 3)
        vel_imu = vel_arr[head_idx]      # (M, 3)
        pos_imu = pos_arr[head_idx]      # (M, 3)
        acc_imu = acc_arr[kp_idx]        # (M, 3) tail acc
        angvel  = gyr_arr[kp_idx]        # (M, 3) tail gyro

        dt_p = pt_offsets - imu_offsets[head_idx]  # (M,)

        # Rodrigues rotation: exp(omega_m * dt_m) applied to each point
        #   rot_q = q*cos + (k×q)*sin + k*(k·q)*(1-cos)
        # where k = omega/|omega|, theta = |omega|*dt
        off_R = imu_state_end.offset_R   # (3, 3) constant
        off_T = imu_state_end.offset_T   # (3,) constant
        R_end_inv = imu_state_end.rot.T  # (3, 3) constant

        # q = off_R @ P_i + off_T for each point
        q = pts_out[:, :3] @ off_R.T + off_T   # (M, 3)

        omega_norms = np.linalg.norm(angvel, axis=1)  # (M,)
        has_rot = omega_norms > 1e-10
        safe_n  = np.where(has_rot, omega_norms, 1.0)
        k = angvel / safe_n[:, None]               # (M, 3) unit axes

        theta   = omega_norms * dt_p               # (M,)
        sin_t   = np.sin(theta)[:, None]           # (M, 1)
        cos_t   = np.cos(theta)[:, None]           # (M, 1)

        k_cross_q = np.cross(k, q)                 # (M, 3)
        k_dot_q   = np.einsum('mi,mi->m', k, q)[:, None]  # (M, 1)

        # exp(omega*dt) @ q for each point
        exp_q = np.where(
            has_rot[:, None],
            q * cos_t + k_cross_q * sin_t + k * k_dot_q * (1.0 - cos_t),
            q,
        )  # (M, 3)

        # R_imu[m] @ exp_q[m]: batched matvec
        R_imu_exp_q = np.einsum('mij,mj->mi', R_imu, exp_q)  # (M, 3)

        # T_ei[m] = pos_imu[m] + vel_imu[m]*dt + 0.5*acc_imu[m]*dt^2 - pos_end
        pos_end = imu_state_end.pos
        T_ei = (pos_imu
                + vel_imu * dt_p[:, None]
                + 0.5 * acc_imu * (dt_p ** 2)[:, None]
                - pos_end)                          # (M, 3)

        combined = R_imu_exp_q + T_ei              # (M, 3)

        # R_end_inv @ combined[m]  →  combined @ R_end_inv.T  (row-vector form)
        R_end_combined = combined @ R_end_inv.T    # (M, 3)

        # off_R.T @ (R_end_combined[m] - off_T)  →  (…) @ off_R  (row-vector form)
        pts_out[:, :3] = (R_end_combined - off_T) @ off_R  # (M, 3)

        return pts_out, state

    # _reset：回到首帧前的初始统计状态（仅在 imu_init 首帧路径中调用）。
    def _reset(self):
        self.mean_acc  = np.array([0.0, 0.0, -1.0])
        self.mean_gyr  = np.zeros(3)
        self.angvel_last = np.zeros(3)
        self.imu_need_init_ = True
        self.start_timestamp_ = -1.0
        self.init_iter_num = 1
        self.last_imu_ = None

    # ------------------------------------------------------------------
    # Process(): top-level driver (mirrors ImuProcess::Process())
    # ------------------------------------------------------------------
    def process(
        self,
        meas: dict,
        state: StateIkfom,
    ) -> tuple[Optional[np.ndarray], StateIkfom]:
        """Drive IMU init and undistortion.

        Returns (pts_undistorted_or_None, updated_state).
        """
        if len(meas.get("imu_stamps", [])) == 0:
            return None, state

        if self.imu_need_init_:
            state = self.imu_init(meas, state)
            self.last_imu_ = {
                "stamp": meas["imu_stamps"][-1],
                "acc":   meas["imu_acc"][-1].copy(),
                "gyro":  meas["imu_gyro"][-1].copy(),
            }
            return None, state

        pts_out, state = self.undistort_pcl(meas, state)
        return pts_out, state
# =============================================================================
# §5  增量式地图 KD-Tree（scipy 实现）  →  见 fastlio_utils.py 末节
#     「增量地图 KD-Tree · FOV 裁剪 · 体↔世界变换」
# -----------------------------------------------------------------------------
# IkdTreeBase / IkdTreeScipy 已下沉至 fastlio_utils.py，见文件头 import。
# =============================================================================
# =============================================================================
# §6  点到平面观测模型 —— 批量平面拟合与 12 维观测雅可比
# -----------------------------------------------------------------------------
# 本节职责：实现迭代 EKF 更新中的观测模型 h_share_model，即由 ikd-Tree 的 5 近邻
# 拟合局部平面、构造点到平面残差 h 及其对状态的观测雅可比 h_x。对应 C++ 版
# laserMapping.cpp / laserMappingOffline.cpp 中的 h_share_model() 函数（该函数
# 在 C++ 中以共享模型回调的形式注入 esekfom::esekf::update_iterated_dyn_share_modified）。
# 本实现为完全向量化的纯 numpy 路径：对 K 个降采样点不做 Python 逐点循环。
#
# 关键公式与数据结构：
#   (1) 平面拟合：对每点的 5 个近邻 nn ∈ R^{5×3}，解超定方程 nn @ n_raw = -1_5，
#       此处经正规方程 (nn^T nn) n_raw = -nn^T 1 用 np.linalg.solve 批量求解；
#       归一化得单位法向 n_hat = n_raw/‖n_raw‖，平面偏置 D = 1/‖n_raw‖，
#       平面质量判据为 5 个近邻到平面的距离均不超过 plane_threshold（0.1 m）。
#   (2) 残差：pd2 = n_hat · p_world + D（有符号点面距），h = -pd2；
#       外点剔除量 s = 1 - 0.9·|pd2|/sqrt(‖p_body‖)，要求 s > 0.9。
#   (3) 雅可比 h_x ∈ R^{N_eff×12}，列块依次对应状态分量
#       [∂h/∂pos(3) | ∂h/∂rot(3) | ∂h/∂offset_R(3) | ∂h/∂offset_T(3)]：
#         ∂h/∂pos      = n_hat
#         ∂h/∂rot      = A = p_imu × (R^T n_hat)
#         ∂h/∂offset_R = B = p_body × (offset_R^T R^T n_hat)（仅在线估计外参时非零）
#         ∂h/∂offset_T = C = R^T n_hat
#
# 计算分解（性能关键设计）：平面拟合只依赖近邻集合、与 EKF 状态无关，故拆出
# fit_planes() 作为“状态无关”部分，由调用方在每帧仅计算一次并在迭代 EKF 的
# 4–5 次 h_share_model 调用间复用（收敛后重新做近邻搜索时才重算）；
# h_share_model() 内仅重复计算“状态相关”部分（坐标变换、门限与雅可比）。
# =============================================================================

NUM_MATCH_POINTS = 5


# -----------------------------------------------------------------------------
# fit_planes：状态无关的批量平面拟合。返回候选点索引、体坐标点、第 5 近邻
# （供距离门限用）、单位法向、平面偏置与平面质量标志，均按候选点（cand 空间）
# 排列。批量 solve 遇到奇异矩阵时逐点回退，保证非奇异点结果与批量解逐位一致。
# -----------------------------------------------------------------------------
def fit_planes(
    pts_body: np.ndarray,       # (K, 3) feats_down_body
    nearest_pts: np.ndarray,    # (K, 5, 3) from ikd-Tree
    nearest_valid: np.ndarray,  # (K,) bool
    plane_threshold: float = 0.1,
):
    """State-INDEPENDENT plane fit over candidate points.

    Everything here is a function of the neighbor sets only — NOT of the EKF
    state — so it can be computed once per scan and reused across the iterated
    EKF update's 4-5 `h_share_model` calls (until the NN set is re-searched on
    convergence). This is what lets the pure-numpy path skip the dominant
    plane-fitting cost on the non-converge iterations.

    Returns a tuple (cand, pb, nn5th, n_hat, D_vec, plane_ok), all in
    cand-space (one row per nearest_valid point):
        cand     (M,)   int   indices into [0..K-1]
        pb       (M,3)        body-frame points (= pts_body[cand])
        nn5th    (M,3)        5th (farthest) neighbor — for the dist gate
        n_hat    (M,3)        unit plane normals
        D_vec    (M,)         plane offsets (1/||n_raw||)
        plane_ok (M,) bool    plane-quality flag (all 5 neighbors within thr)

    NOTE: the plane fit is computed for ALL candidate points here, whereas the
    legacy inline path fit only the dist-gated subset. Per-point results are
    identical (each normal depends only on that point's 5 neighbors); the only
    difference is fitting a few points that the dist gate later drops, which is
    amortised away by reuse across iterations.
    """
    cand = np.where(nearest_valid)[0]
    M = len(cand)
    if M == 0:
        z = np.zeros((0, 3))
        return cand, z, z, z, np.zeros(0), np.zeros(0, dtype=bool)

    pb = pts_body[cand]          # (M, 3)
    nn = nearest_pts[cand]       # (M, 5, 3)

    # Plane fit: solve A @ n = -ones(5) via normal equations A^T A n = A^T b.
    ATA = np.einsum('mij,mik->mjk', nn, nn)   # (M, 3, 3)
    ATb = -nn.sum(axis=1)                       # (M, 3)
    try:
        n_raw = np.linalg.solve(ATA, ATb)       # (M, 3)
    except np.linalg.LinAlgError:
        # A single singular ATA would otherwise poison the whole batch. Fitting
        # for ALL candidates (not just the dist-gated subset, as the legacy path
        # did) makes singular far-points more likely, so fall back PER POINT:
        # every non-singular point still goes through np.linalg.solve — which is
        # bit-identical to the batched solve for that matrix — so selected points
        # match the legacy result exactly; only genuinely singular points (which
        # fail the plane-quality check below anyway) take the lstsq path.
        b5 = -np.ones(5)
        n_raw = np.empty((M, 3))
        for _j in range(M):
            try:
                n_raw[_j] = np.linalg.solve(ATA[_j], ATb[_j])
            except np.linalg.LinAlgError:
                n_raw[_j] = np.linalg.lstsq(nn[_j], b5, rcond=None)[0]

    norms   = np.linalg.norm(n_raw, axis=1)              # (M,)
    norm_ok = norms > 1e-10
    safe_n  = np.where(norms[:, None] > 1e-10, norms[:, None], 1.0)
    n_hat   = n_raw / safe_n                              # (M, 3)
    D_vec   = np.where(norms > 1e-10, 1.0 / norms, 0.0)   # (M,)

    dists    = np.abs(np.einsum('mpd,md->mp', nn, n_hat) + D_vec[:, None])  # (M, 5)
    plane_ok = norm_ok & (~np.any(dists > plane_threshold, axis=1))         # (M,)

    nn5th = nn[:, NUM_MATCH_POINTS - 1, :]    # (M, 3)
    return cand, pb, nn5th, n_hat, D_vec, plane_ok


# -----------------------------------------------------------------------------
# h_share_model：观测模型主函数。给定当前状态与近邻集合，输出观测雅可比 h_x
# （N_eff×12）、残差 h（负的有符号点面距）与有效点掩码 valid_mask。planes 参数
# 用于传入 fit_planes() 的每帧缓存；未提供时（单元测试或收敛后近邻集变化的
# 重搜索场合）在函数内部现算。各门限（第 5 近邻距离平方 ≤ 5、体坐标范数、
# s > 0.9）与 C++ 版逐一对应，最终有效集为各判据的交集（与串行顺序等价）。
# -----------------------------------------------------------------------------
def h_share_model(
    state: StateIkfom,
    pts_body: np.ndarray,       # (K, 3) feats_down_body
    nearest_pts: np.ndarray,    # (K, 5, 3) from ikd-Tree
    nearest_valid: np.ndarray,  # (K,) bool — which points have valid NN
    extrinsic_est: bool = True,
    plane_threshold: float = 0.1,
    planes=None,                # optional precomputed fit_planes() result
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute h_x (Jacobian) and h (residuals) for EKF update.

    Fully vectorised version — no Python loop over K points.

    Args:
        state:          current EKF state
        pts_body:       (K, 3) points in LiDAR body frame
        nearest_pts:    (K, 5, 3) nearest-neighbor world points from ikd-Tree
        nearest_valid:  (K,) bool — points with valid NN search
        extrinsic_est:  whether to estimate LiDAR-IMU extrinsics
        plane_threshold: plane-fitting residual threshold

    Returns:
        h_x:        (N_eff, 12) measurement Jacobian
        h:          (N_eff,)   residuals (negative plane distances)
        valid_mask: (K,) bool  which input points were effective
    """
    K = len(pts_body)

    rot_R = state.rot        # (3, 3)
    pos_p = state.pos        # (3,)
    off_R = state.offset_R   # (3, 3)
    off_T = state.offset_T   # (3,)

    # --- Pure numpy reference path ---
    valid_mask = np.zeros(K, dtype=bool)

    # State-independent plane fit (candidates, normals, plane-quality). Computed
    # once per scan by the caller and reused across the iterated-EKF iterations;
    # only recomputed here if not supplied (unit tests, or the converge
    # re-search where the NN set changed). See fit_planes() for the cache layout.
    if planes is None:
        planes = fit_planes(pts_body, nearest_pts, nearest_valid, plane_threshold)
    cand, pb, nn5th, n_hat, D_vec, plane_ok = planes
    M = len(cand)
    if M == 0:
        return np.zeros((0, 12)), np.zeros(0), valid_mask

    # ---- State-DEPENDENT part (recomputed every EKF iteration) ----
    # Transform candidate points to the world frame with the current state:
    #   pts_imu   = off_R @ pb + off_T ;  pts_world = rot_R @ pts_imu + pos_p
    pts_imu   = pb @ off_R.T + off_T       # (M, 3)
    pts_world = pts_imu @ rot_R.T + pos_p  # (M, 3)

    # Distance-to-5th-NN gate (mirrors pointSearchSqDis[4] > 5).
    diff_5th = nn5th - pts_world                                   # (M, 3)
    dist_ok  = np.einsum('mi,mi->m', diff_5th, diff_5th) <= 5.0    # (M,) bool

    # Plane distance + point-quality (s) outlier rejection.
    pd2      = np.einsum('md,md->m', pts_world, n_hat) + D_vec     # (M,)
    pb_norms = np.linalg.norm(pb, axis=1)                          # (M,)
    pbody_ok = pb_norms > 1e-6
    safe_pbn = np.where(pbody_ok, pb_norms, 1.0)
    s_val    = np.where(pbody_ok,
                        1.0 - 0.9 * np.abs(pd2) / np.sqrt(safe_pbn),
                        -1.0)
    s_ok     = s_val > 0.9

    # Effective points = plane-quality (cached) ∧ dist ∧ body-norm ∧ s.
    # (Same final set as the legacy order dist→fit→quality; AND is commutative.)
    all_ok = plane_ok & dist_ok & pbody_ok & s_ok    # (M,) bool
    valid_mask[cand[all_ok]] = True

    N_eff = int(all_ok.sum())
    if N_eff == 0:
        return np.zeros((0, 12)), np.zeros(0), valid_mask

    # ---- Jacobian (vectorised over the N_eff effective points) ----
    eff_n_hat = n_hat[all_ok]        # (N_eff, 3)
    eff_imu   = pts_imu[all_ok]      # (N_eff, 3) = off_R @ pb + off_T
    eff_pb    = pb[all_ok]           # (N_eff, 3) body-frame
    eff_pd2   = pd2[all_ok]          # (N_eff,)

    # C_vec[m] = rot_R^T @ n_hat[m]  → batch: n_hat @ rot_R  (row-vector convention)
    C_vec = eff_n_hat @ rot_R            # (N_eff, 3)
    # A_vec[m] = hat(pts_imu[m]) @ C_vec[m] = pts_imu[m] × C_vec[m]
    A_vec = np.cross(eff_imu, C_vec)     # (N_eff, 3)

    if extrinsic_est:
        # B_vec[m] = pb[m] × (off_R^T @ C_vec[m]) = pb[m] × (C_vec[m] @ off_R)
        offR_T_C = C_vec @ off_R         # (N_eff, 3)
        B_vec    = np.cross(eff_pb, offR_T_C)  # (N_eff, 3)
        h_x = np.concatenate([eff_n_hat, A_vec, B_vec, C_vec], axis=1)  # (N_eff, 12)
    else:
        zeros = np.zeros_like(A_vec)
        h_x = np.concatenate([eff_n_hat, A_vec, zeros, zeros], axis=1)  # (N_eff, 12)

    h = -eff_pd2     # (N_eff,) residuals
    return h_x, h, valid_mask
# ============================================================================
# §7  rosbag 裸字节解析（raw-bytes message parsers）
# ----------------------------------------------------------------------------
# 本节职责：绕过 rospy 的消息反序列化层，直接把 rosbag 中的裸字节 payload 解析为
# 流水线所需的 numpy 数组。对应 C++ 版的两处边界：preprocess.cpp 中按消息类型
# 展开点云/IMU 字段的解析层，以及 laserMappingOffline.cpp 中 rosbag::View 顺序
# 读包的 I/O 边界。
#
# 动机：rosbag.Bag.read_messages(raw=True) 返回原始字节而非构造完整 Python 消息
# 对象，单条消息迭代开销由 ~4 ms 降至 ~0.022 ms（约 200 倍）。本节据 ROS1 序列
# 化格式（小端、变长数组带 uint32 长度前缀、定长数组无前缀）手工解包。
#
# 支持的消息类型：
#   - sensor_msgs/Imu            → parse_imu_bytes()：header 之后为 37 个连续
#     float64（四元数 4 + 两个 3×1 向量 + 三个 9 元协方差 = 4+6+27），仅取角速度
#     与线加速度；
#   - livox_ros_driver/CustomMsg → parse_livox_bytes()：逐点结构 CustomPoint 为
#     offset_time(u32) + x/y/z(f32) + reflectivity/tag/line(u8)，共 19 字节且成员
#     间无填充（_LIVOX_POINT_DTYPE 的 itemsize == 19 由 assert 保证），可用
#     np.frombuffer 零拷贝批量视图化。
#
# 关键精度约定：时间戳换算必须写作 sec + nsec / 1e9（而非乘以 1e-9），以与
# rospy Time.to_sec() 逐位一致；详见 _parse_header 的注释。非 Livox 的
# PointCloud2 包在 §8 主循环中经 rospy 反序列化后由 pcl2_to_array() 处理。
#
# 本节符号（_LIVOX_POINT_DTYPE / _parse_header / parse_imu_bytes /
# parse_livox_bytes，以及 §8 中的 _try_import_rosbag / imu_msg_to_dict /
# _LIVOX_GETTER / livox_msg_to_array / pcl2_to_array）已下沉至
# fastlio_utils.py §B，见文件头 import。
# ============================================================================
# =============================================================================
# §8  离线主流程（对应 C++ 版 src/laserMappingOffline.cpp）
# -----------------------------------------------------------------------------
# 本节职责：
#   读取 rosbag，在纯 Python/numpy 环境下驱动完整的 FAST-LIO2 离线建图流水线，
#   即 C++ 离线节点中 "for (const rosbag::MessageInstance &m : view)" 主循环的
#   逐语句对应实现，并输出：
#     - PCD 地图：   <output_dir>/PCD/map_offline_py.pcd
#     - TUM 轨迹：   <output_dir>/Log/trajectory_py_tum.txt
#       （每行格式：timestamp tx ty tz qx qy qz qw）
#
# 每帧扫描的处理流程（与 C++ 主循环一一对应）：
#   rosbag 原始字节流
#     → parse_livox_bytes()/parse_imu_bytes()   消息解析（raw=True 绕过 rospy）
#     → _sync_packages()                        LiDAR 帧与 IMU 时间窗配对
#     → p_imu.process()                         IMU 初始化 + 运动畸变校正（§4）
#     → voxel_downsample()                      体素栅格降采样（filter_size_surf）
#     → lasermap_fov_segment()                  按探测半径裁剪局部地图（移动门控）
#     → Nearest_Search() + fit_planes()         5-NN 平面拟合（§5、§6）
#     → update_iterated()                       SO(3)×S2 流形上的迭代误差状态 EKF（§3）
#     → map_incremental()                       新点按体素判据增量插入 ikd-Tree
#     → 轨迹记录 / 全分辨率世界系点云累积
#
# 关键数据结构：
#   - measures 字典：{"lidar_pts", "lidar_beg", "lidar_end",
#                     "imu_stamps", "imu_acc", "imu_gyro"}，
#     对应 C++ 的 MeasureGroup（common_lib.h）。
#   - 点云统一为 (N,4) float32 数组 [x, y, z, offset_ms]，
#     第 4 列为该点相对扫描起始时刻的偏移（毫秒），对应 PCL 点的 curvature 字段。
#   - PhaseTimer：--profile 时的分区域计时器；未开启时以 _NullTimer 零开销替身。
#
# 用法示例（原模块 docstring，改写为注释保留）：
#   # Livox AVIA 包：
#   python3 offline_main.py --bag your.bag --config config/avia.yaml \
#       --output_dir /root/catkin_ws/src/FAST_LIO/
#   # Velodyne 包：
#   python3 offline_main.py --bag your.bag --config config/velodyne.yaml \
#       --lidar_type 2 --output_dir /root/catkin_ws/src/FAST_LIO/
#   依赖：rosbag（ROS Noetic，函数内惰性导入）、numpy、scipy、open3d（可选）。
# =============================================================================


# PhaseTimer / _TimerCtx / _NullTimer / _NullCtx / _NULL_CTX / _NULL_TIMER
# 已下沉至 fastlio_utils.py §C，见文件头 import。

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INIT_TIME      = 0.1   # seconds before EKF is considered initialized
LASER_PT_COV   = 0.001
FILTER_SIZE_MAP = 0.5
LIDAR_AVIA     = 1
LIDAR_VELODYNE = 2
LIDAR_OUSTER   = 3
LIDAR_MARSIM   = 4


# load_config / get_param（配置加载）与 ROS 消息 helpers
# （_try_import_rosbag / imu_msg_to_dict / _LIVOX_GETTER / livox_msg_to_array /
# pcl2_to_array）已下沉至 fastlio_utils.py §B、§D，见文件头 import。
# voxel_downsample（体素栅格降采样）已下沉至 fastlio_utils.py §F。


# point_body_to_world（体→世界变换）与 lasermap_fov_segment（FOV 球外裁剪）
# 已下沉至 fastlio_utils.py 末节「增量地图 KD-Tree · FOV 裁剪 · 体↔世界变换」，
# 见文件头 import。


# ---------------------------------------------------------------------------
# map_incremental (mirrors C++ map_incremental())
# ---------------------------------------------------------------------------
# 讲解：体素判据的向量化实现——当点比其最近邻更接近所在体素中心（且最近邻
# 未远离该体素）时才插入，需降采样/免降采样两类点分别调用 Add_Points。
def map_incremental(
    ikdtree: IkdTreeScipy,
    feats_down_world: np.ndarray,   # (K, 3)
    nearest_pts: np.ndarray,        # (K, 5, 3)
    nearest_valid: np.ndarray,      # (K,) bool
    filter_map: float = 0.5,
    flg_inited: bool = True,
) -> None:
    """Add new points to the ikd-Tree with downsampling (vectorised)."""
    K = len(feats_down_world)
    pts = feats_down_world  # (K, 3)

    # Points without valid NN or before init: always add with downsampling
    uninit_mask = ~(nearest_valid & flg_inited)   # (K,)
    init_mask = ~uninit_mask

    # Arrays of indices into pts for the two "to-add" categories. Kept as
    # numpy arrays (no list() conversions) and concatenated at the end.
    to_add_chunks: List[np.ndarray] = [np.where(uninit_mask)[0]]
    no_need_down_idx: np.ndarray = np.empty(0, dtype=np.intp)

    if init_mask.any():
        idx_i = np.where(init_mask)[0]           # (N_init,)
        p_i   = pts[idx_i]                       # (N_init, 3)
        nn_i  = nearest_pts[idx_i]               # (N_init, 5, 3)

        half  = 0.5 * filter_map
        mid   = np.floor(p_i / filter_map) * filter_map + half  # (N_init, 3)
        dist_p = np.linalg.norm(p_i - mid, axis=1)               # (N_init,)

        # "no-need-down" if nearest[0] is far from mid in ALL three dims
        nn0 = nn_i[:, 0, :]
        far = np.all(np.abs(nn0 - mid) > half, axis=1)
        no_need_down_idx = idx_i[far]

        near_mask = ~far
        if near_mask.any():
            idx_near = idx_i[near_mask]
            nn_near  = nn_i[near_mask]
            mid_near = mid[near_mask]
            dp_near  = dist_p[near_mask]

            nn_to_mid = np.linalg.norm(nn_near - mid_near[:, None, :], axis=2)
            need_add = np.all(nn_to_mid >= dp_near[:, None], axis=1)
            to_add_chunks.append(idx_near[need_add])

    to_add_idx = np.concatenate(to_add_chunks) if len(to_add_chunks) > 1 else to_add_chunks[0]
    all_to_add  = pts[to_add_idx]    if to_add_idx.size    else np.zeros((0, 3))
    all_no_down = pts[no_need_down_idx] if no_need_down_idx.size else np.zeros((0, 3))

    if len(all_to_add) > 0:
        ikdtree.Add_Points(all_to_add, downsample=True)
    if len(all_no_down) > 0:
        ikdtree.Add_Points(all_no_down, downsample=False)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
# 讲解：run() 即离线主循环。内部闭包 _sync_packages() 对应 C++
# sync_packages()：以 LiDAR 帧尾时刻（帧首 + 末点偏移，滑动平均估计扫描周期）
# 为界收集 IMU 窗口。每帧依次执行 IMU 去畸变 → FOV 裁剪 → 降采样 →
# 迭代 EKF 更新 → 地图增量插入 → 轨迹/点云记录。
def run(
    bag_path: str,
    lid_topic: str,
    imu_topic: str,
    output_dir: str,
    lidar_type: int = LIDAR_AVIA,
    filter_size_surf: float = 0.5,
    filter_size_map: float  = 0.5,
    det_range: float        = 300.0,
    gyr_cov: float          = 0.1,
    acc_cov: float          = 0.1,
    b_gyr_cov: float        = 0.0001,
    b_acc_cov: float        = 0.0001,
    extrinsic_T: np.ndarray = None,
    extrinsic_R: np.ndarray = None,
    extrinsic_est_en: bool  = True,
    point_filter_num: int   = 2,
    max_iterations: int     = 3,   # matches C++ launch param max_iteration
    num_max_init: int       = 10,
    max_scans: int          = 0,
    profile: bool           = False,
) -> None:
    rosbag = _try_import_rosbag()

    os.makedirs(os.path.join(output_dir, "PCD"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Log"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)

    # ------------------------------------------------------------------ init
    state = StateIkfom()
    p_imu = ImuProcess()
    p_imu.lidar_type = lidar_type
    if extrinsic_T is not None:
        p_imu.set_extrinsic(extrinsic_T, extrinsic_R if extrinsic_R is not None else np.eye(3))
    p_imu.cov_gyr      = np.full(3, gyr_cov)
    p_imu.cov_acc      = np.full(3, acc_cov)
    p_imu.cov_bias_gyr = np.full(3, b_gyr_cov)
    p_imu.cov_bias_acc = np.full(3, b_acc_cov)
    p_imu._MAX_INI_COUNT = num_max_init

    ikdtree = IkdTreeScipy()
    ikdtree.set_downsample_param(filter_size_map)

    # Profiling collector (no-op when --profile is off)
    _timer: Any = PhaseTimer() if profile else _NULL_TIMER

    # Movement gate for lasermap_fov_segment: the sphere radius is
    # det_range * mov_threshold (e.g. 450 * 1.5 = 675m). Points only leave
    # the sphere when the LiDAR has moved a significant fraction of that
    # radius. We track the last position where fov_segment was actually
    # run and gate subsequent calls on a conservative threshold.
    last_fov_pos: Optional[np.ndarray] = None
    fov_gate_dist = det_range * 0.25  # conservative: run fov_segment every ~det_range/4 of motion

    lidar_buffer: deque = deque()
    time_buffer:  deque = deque()
    imu_buffer:   deque = deque()

    trajectory: List[dict] = []
    # Per-frame map output: each scan's undistorted (LiDAR-body) cloud is
    # streamed to one binary file (frames/clouds.bin) and its estimated pose +
    # online-estimated extrinsics recorded (frames/index.npz). The dense world
    # map is reconstructed on demand by fastlio_utils.aggregate_map() — i.e.
    # `python3 fastlio_utils.py --output_dir <dir>`. Keeps the run memory-light:
    # no whole-map accumulation in RAM, each write is just the current frame.
    _clouds_f = open(os.path.join(output_dir, "frames", "clouds.bin"), "wb")
    fr_count: List[int]        = []   # points per frame
    fr_pos:   List[np.ndarray] = []   # (3,)   state.pos      per frame
    fr_rot:   List[np.ndarray] = []   # (3,3)  state.rot      per frame
    fr_offR:  List[np.ndarray] = []   # (3,3)  state.offset_R per frame (EKF-estimated)
    fr_offT:  List[np.ndarray] = []   # (3,)   state.offset_T per frame
    fr_stamp: List[float]      = []   # scan-end timestamp per frame

    flg_first_scan   = True
    flg_EKF_inited   = False
    lidar_pushed     = False
    lidar_end_time   = 0.0
    first_lidar_time = 0.0
    lidar_mean_scantime = 0.0
    scan_num = 0
    last_timestamp_imu = -1.0
    last_timestamp_lidar = 0.0

    # ------------------------------------------------------------------ bag
    print(f"Opening bag: {bag_path}")
    bag = rosbag.Bag(bag_path)
    topics = [lid_topic, imu_topic]
    # raw=True returns (msgtype, bytes, md5, (sec,nsec), msgtype_class) instead of
    # an instantiated rospy message object. The iterator cost drops ~200x — the
    # actual field extraction happens in our parse_*_bytes() helpers, giving full
    # control over allocations.
    view = bag.read_messages(topics=topics, raw=True)
    total = bag.get_message_count(topic_filters=topics)

    t_start = time.time()
    msg_count = 0
    lidar_count = 0

    def _sync_packages():
        nonlocal lidar_pushed, lidar_end_time, lidar_mean_scantime, scan_num
        if not lidar_buffer or not imu_buffer:
            return None

        if not lidar_pushed:
            pts_cloud = lidar_buffer[0]
            t_beg     = time_buffer[0]

            if len(pts_cloud) <= 1:
                lidar_end_time = t_beg + lidar_mean_scantime
            elif pts_cloud[-1, 3] / 1000.0 < 0.5 * lidar_mean_scantime:
                lidar_end_time = t_beg + lidar_mean_scantime
            else:
                scan_num += 1
                dt_scan  = pts_cloud[-1, 3] / 1000.0
                lidar_end_time = t_beg + dt_scan
                lidar_mean_scantime += (dt_scan - lidar_mean_scantime) / scan_num

            if lidar_type == LIDAR_MARSIM:
                lidar_end_time = t_beg

            lidar_pushed = True

        if last_timestamp_imu < lidar_end_time:
            return None

        # Collect IMU in window
        imu_win = []
        while imu_buffer and imu_buffer[0]["stamp"] <= lidar_end_time:
            imu_win.append(imu_buffer.popleft())

        pts = lidar_buffer.popleft()
        t   = time_buffer.popleft()
        lidar_pushed = False

        return {
            "lidar_pts":  pts,
            "lidar_beg":  t,
            "lidar_end":  lidar_end_time,
            "imu_stamps": np.array([m["stamp"] for m in imu_win]),
            "imu_acc":    np.array([m["acc"]   for m in imu_win]),
            "imu_gyro":   np.array([m["gyro"]  for m in imu_win]),
        }

    # ------------------------------------------------------------------ loop
    # Raw tuple from rosbag.Bag.read_messages(raw=True):
    #   (msgtype_str, raw_bytes, md5, (sec, nsec), msgtype_class)
    # For Livox + IMU we skip rospy altogether. For PointCloud2 we fall back
    # to deserializing via msgtype_class — still faster than the default iterator
    # because we only pay the rospy cost on LiDAR messages (IMU is ~20x more common).
    _bag_done = False
    for topic, raw_tuple, t_msg in view:
        if _bag_done:
            break
        msg_count += 1
        if msg_count % 500 == 0:
            elapsed = time.time() - t_start
            pct = msg_count / total * 100.0
            print(f"  [{pct:5.1f}%] {msg_count}/{total} msgs  {elapsed:.1f}s  "
                  f"poses={len(trajectory)}", end="\r", flush=True)

        raw_bytes = raw_tuple[1]

        if topic == lid_topic:
            pts = None
            stamp = 0.0
            with _timer.region("bag_parse_lidar"):
                if lidar_type == LIDAR_AVIA:
                    try:
                        stamp, pts = parse_livox_bytes(raw_bytes, point_filter_num)
                    except Exception:
                        pts = None
                if pts is None:
                    # Non-Livox (Velodyne/Ouster): deserialize via msgtype class.
                    # Still faster than rospy iterator because we bypass object
                    # construction for all non-lidar messages.
                    try:
                        msg = raw_tuple[4]().deserialize(raw_bytes)
                        pts = pcl2_to_array(msg, point_filter_num, lidar_type)
                        stamp = msg.header.stamp.to_sec()
                    except Exception:
                        pts = None
            if pts is not None and len(pts) > 0:
                if stamp < last_timestamp_lidar:
                    lidar_buffer.clear()
                    time_buffer.clear()
                last_timestamp_lidar = stamp
                lidar_buffer.append(pts)
                time_buffer.append(stamp)
                lidar_count += 1

        elif topic == imu_topic:
            with _timer.region("bag_parse_imu"):
                imu = parse_imu_bytes(raw_bytes)
            if imu["stamp"] < last_timestamp_imu:
                imu_buffer.clear()
            last_timestamp_imu = imu["stamp"]
            imu_buffer.append(imu)

        # Process all ready scan-IMU pairs
        while True:
            meas = _sync_packages()
            if meas is None:
                break

            if flg_first_scan:
                first_lidar_time = meas["lidar_beg"]
                p_imu.first_lidar_time = first_lidar_time
                flg_first_scan = False
                continue

            # IMU undistortion
            with _timer.region("imu_undistort"):
                pts_undistort, state = p_imu.process(meas, state)

            if pts_undistort is None or len(pts_undistort) == 0:
                continue

            flg_EKF_inited = (meas["lidar_beg"] - first_lidar_time) >= INIT_TIME

            # FOV segment (movement-gated). The sphere of "keep" points has
            # radius det_range*mov_threshold (~675 m by default); no point
            # can leave the sphere unless the LiDAR has moved a comparable
            # distance. Gate avoids calling flatten_points() every scan.
            pos_lid = state.pos + state.rot @ state.offset_T
            with _timer.region("fov_segment"):
                run_fov = (last_fov_pos is None or
                           float(np.linalg.norm(pos_lid - last_fov_pos)) >
                           fov_gate_dist)
                if run_fov:
                    lasermap_fov_segment(ikdtree, pos_lid, det_range)
                    last_fov_pos = pos_lid.copy()

            # Voxel downsample
            with _timer.region("voxel_surf"):
                feats_down = voxel_downsample(pts_undistort, filter_size_surf)
            feats_down_size = len(feats_down)

            if feats_down_size < 5:
                continue

            feats_down_body  = feats_down[:, :3]
            with _timer.region("point_transforms"):
                feats_down_world = point_body_to_world(state, feats_down)[:, :3]

            # Build tree if first scan
            if ikdtree.Root_Node is None:
                if feats_down_size > 5:
                    ikdtree.set_downsample_param(filter_size_map)
                    ikdtree.Build(feats_down_world)
                continue

            # Nearest-neighbor search
            with _timer.region("nn_search_init"):
                nn_pts_all, sq_dists_all = ikdtree.Nearest_Search(
                    feats_down_world, NUM_MATCH_POINTS)
            # valid: has enough neighbors and last neighbor close enough
            nn_valid = (sq_dists_all[:, -1] <= 5.0)

            # State propagated reference (for boxminus)
            state_prop = state.copy()

            # Precompute the state-independent plane fit ONCE per scan. The
            # iterated EKF calls _h_share 4-5× with the same NN set on the
            # non-converge iterations, so reusing this cache skips the dominant
            # plane-fitting cost on all but the converge re-search.
            planes_cache = fit_planes(feats_down_body, nn_pts_all, nn_valid)

            # h_share closure for update_iterated
            def _h_share(s: StateIkfom, converge: bool):
                # Re-transform points to world frame with current state s
                fw = (s.rot @ (s.offset_R @ feats_down_body.T + s.offset_T[:, None])
                      + s.pos[:, None]).T

                # NN search (re-use results if converged). On converge the NN
                # set changes, so the cached plane fit is invalid → refit.
                if converge:
                    nn_p, sq_d = ikdtree.Nearest_Search(fw, NUM_MATCH_POINTS)
                    nv = (sq_d[:, -1] <= 5.0)
                    planes = fit_planes(feats_down_body, nn_p, nv)
                else:
                    nn_p = nn_pts_all
                    nv   = nn_valid
                    planes = planes_cache

                h_x, h, _ = h_share_model(
                    s,
                    feats_down_body,
                    nn_p,
                    nv,
                    extrinsic_est=extrinsic_est_en,
                    planes=planes,
                )
                return h_x, h, (len(h) > 0)

            # Iterated EKF update
            with _timer.region("ekf_update"):
                state = update_iterated(
                    state, state_prop, _h_share,
                    R_scalar=LASER_PT_COV,
                    max_iter=max_iterations,
                )

            # Update nearest-neighbors using final state
            with _timer.region("point_transforms"):
                feats_down_world = point_body_to_world(state, feats_down)[:, :3]
            with _timer.region("nn_search_final"):
                nn_pts_final, sq_dists_final = ikdtree.Nearest_Search(
                    feats_down_world, NUM_MATCH_POINTS)
            nn_valid_final = (sq_dists_final[:, -1] <= 5.0)

            # Map update
            with _timer.region("map_incremental"):
                map_incremental(ikdtree, feats_down_world, nn_pts_final,
                                nn_valid_final, filter_size_map, flg_EKF_inited)

            # Per-frame map output: stream this scan's undistorted body-frame
            # cloud + record its pose & extrinsics (see setup above). The dense
            # world map is reconstructed later by aggregate_map() applying
            # exactly point_body_to_world per frame.
            with _timer.region("map_accum_full"):
                xyz = np.ascontiguousarray(pts_undistort[:, :3], dtype=np.float32)
                _clouds_f.write(xyz.tobytes())
                fr_count.append(len(xyz))
                fr_pos.append(state.pos.copy())
                fr_rot.append(state.rot.copy())
                fr_offR.append(state.offset_R.copy())
                fr_offT.append(state.offset_T.copy())
                fr_stamp.append(lidar_end_time)

            # Record trajectory (TUM: t tx ty tz qx qy qz qw)
            with _timer.region("traj_record"):
                q = rot_to_quat(state.rot)
                trajectory.append({
                    "t":  lidar_end_time,
                    "tx": state.pos[0], "ty": state.pos[1], "tz": state.pos[2],
                    "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3],
                })

            if max_scans > 0 and len(trajectory) >= max_scans:
                _bag_done = True
                break

    bag.close()
    elapsed = time.time() - t_start
    print(f"\nDone. Poses={len(trajectory)}, wall time={elapsed:.1f}s")

    # ------------------------------------------------------------------ save
    save_t0 = time.perf_counter()
    _save_trajectory(trajectory, os.path.join(output_dir, "Log", "trajectory_py_tum.txt"))
    _clouds_f.close()
    # Per-frame index: pose + online-estimated extrinsics + point count per
    # frame, consumed by fastlio_utils.aggregate_map() to rebuild the world map.
    np.savez(os.path.join(output_dir, "frames", "index.npz"),
             count=np.asarray(fr_count, dtype=np.int64),
             pos=np.asarray(fr_pos, dtype=np.float64),
             rot=np.asarray(fr_rot, dtype=np.float64),
             offset_R=np.asarray(fr_offR, dtype=np.float64),
             offset_T=np.asarray(fr_offT, dtype=np.float64),
             stamp=np.asarray(fr_stamp, dtype=np.float64))
    save_time = time.perf_counter() - save_t0
    print(f"Per-frame map -> {os.path.join(output_dir, 'frames')} "
          f"({len(fr_count)} frames). Aggregate + visualize:\n"
          f"  python3 {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fastlio_utils.py')} "
          f"--output_dir {output_dir}")
    if profile:
        _timer.record("output_save", save_time)

    if profile:
        print("\n=== Profile report (per-region timing) ===")
        print(_timer.report(elapsed + save_time))
        # Pure-compute vs I/O breakdown
        compute_regions = ["imu_undistort", "fov_segment", "voxel_surf",
                           "nn_search_init", "ekf_update", "nn_search_final",
                           "map_incremental", "point_transforms"]
        io_regions      = ["bag_parse_lidar", "bag_parse_imu",
                           "map_accum_full", "traj_record", "output_save"]
        compute_total = sum(_timer._totals.get(r, 0.0) for r in compute_regions)
        io_total      = sum(_timer._totals.get(r, 0.0) for r in io_regions)
        other         = (elapsed + save_time) - compute_total - io_total
        print(f"\n  SLAM compute : {compute_total:8.3f} s "
              f"({compute_total/(elapsed+save_time)*100:5.1f}%)")
        print(f"  I/O + output : {io_total:8.3f} s "
              f"({io_total/(elapsed+save_time)*100:5.1f}%)")
        print(f"  Other/glue   : {other:8.3f} s "
              f"({other/(elapsed+save_time)*100:5.1f}%)")


# _save_trajectory / _save_pcd（文件输出）已下沉至 fastlio_utils.py §E，
# _build_arg_parser（命令行解析）已下沉至 fastlio_utils.py §D，见文件头 import。


def main():
    args = _build_arg_parser().parse_args()

    # Load config if provided
    cfg = {}
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)

    lid_topic  = get_param(cfg, "common", "lid_topic",  default=args.lid_topic)
    imu_topic  = get_param(cfg, "common", "imu_topic",  default=args.imu_topic)
    lidar_type = get_param(cfg, "preprocess", "lidar_type", default=args.lidar_type)

    ext_T_list = get_param(cfg, "mapping", "extrinsic_T", default=None)
    ext_R_list = get_param(cfg, "mapping", "extrinsic_R", default=None)
    ext_T = np.array(ext_T_list) if ext_T_list else None
    ext_R = np.array(ext_R_list).reshape(3, 3) if ext_R_list else None
    extrinsic_est_en = bool(get_param(cfg, "mapping", "extrinsic_est_en", default=True))

    gyr_cov   = get_param(cfg, "mapping", "gyr_cov",   default=0.1)
    acc_cov   = get_param(cfg, "mapping", "acc_cov",   default=0.1)
    b_gyr_cov = get_param(cfg, "mapping", "b_gyr_cov", default=0.0001)
    b_acc_cov = get_param(cfg, "mapping", "b_acc_cov", default=0.0001)

    run(
        bag_path         = args.bag,
        lid_topic        = lid_topic,
        imu_topic        = imu_topic,
        output_dir       = args.output_dir,
        lidar_type       = int(lidar_type),
        filter_size_surf = args.filter_surf,
        filter_size_map  = args.filter_map,
        gyr_cov          = float(gyr_cov),
        acc_cov          = float(acc_cov),
        b_gyr_cov        = float(b_gyr_cov),
        b_acc_cov        = float(b_acc_cov),
        extrinsic_T      = ext_T,
        extrinsic_R      = ext_R,
        extrinsic_est_en = extrinsic_est_en,
        point_filter_num = args.point_filter_num,
        max_iterations   = args.max_iter,
        max_scans        = args.max_scans,
        profile          = args.profile,
    )


if __name__ == "__main__":
    main()
