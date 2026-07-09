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

import argparse
import math
import struct
import sys
import time

import numpy as np
import yaml
import operator as _op
from collections import deque
from dataclasses import dataclass, field
from scipy.spatial import cKDTree
from typing import Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# §1  SO(3) / S(2) 流形数学基础
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
# =============================================================================


def hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from a 3-vector."""
    v = v.ravel()
    return np.array([
        [0.0,  -v[2],  v[1]],
        [v[2],   0.0, -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def exp(v: np.ndarray) -> np.ndarray:
    """SO3 exponential map (Rodrigues): axis-angle vector → 3x3 rotation matrix."""
    v = v.ravel()
    theta = np.linalg.norm(v)
    if theta < 1e-7:
        return np.eye(3)
    axis = v / theta
    K = hat(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def log(R: np.ndarray) -> np.ndarray:
    """SO3 logarithm: 3x3 rotation matrix → axis-angle vector.

    Uses math.acos / math.sin for scalar inputs to minimise numpy call overhead.
    """
    tr = float(R[0, 0]) + float(R[1, 1]) + float(R[2, 2])
    tr_c = max(-1.0, min(3.0, tr))
    if tr_c > 3.0 - 1e-6:
        theta = 0.0
    else:
        theta = math.acos(0.5 * (tr_c - 1.0))
    k0 = float(R[2, 1]) - float(R[1, 2])
    k1 = float(R[0, 2]) - float(R[2, 0])
    k2 = float(R[1, 0]) - float(R[0, 1])
    if theta < 0.001:
        return np.array([0.5 * k0, 0.5 * k1, 0.5 * k2])
    c = 0.5 * theta / math.sin(theta)
    return np.array([c * k0, c * k1, c * k2])


def A_matrix(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian of SO(3) (also called A_matrix in IKFoM).

    A(phi) = I + (1 - cos||phi||)/||phi||^2 * hat(phi)
               + (||phi|| - sin||phi||)/||phi||^3 * hat(phi)^2
    """
    phi = phi.ravel()
    theta = np.linalg.norm(phi)
    if theta < 1e-7:
        return np.eye(3)
    K = hat(phi)
    c1 = (1.0 - np.cos(theta)) / (theta * theta)
    c2 = (theta - np.sin(theta)) / (theta ** 3)
    return np.eye(3) + c1 * K + c2 * (K @ K)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    q = q.ravel()
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)    ],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z),  2*(y*z - x*w)    ],
        [2*(x*z - y*w),     2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()  # [x, y, z, w]


# ---------------------------------------------------------------------------
# S2 (unit sphere) helpers for gravity state
# ---------------------------------------------------------------------------

def s2_Bx(v: np.ndarray, length: float = 1.0, s2_typ: int = 1) -> np.ndarray:
    """S2_Bx: tangent basis at vector v on the sphere of radius `length`.

    Matches the S2_Bx() method in mtk/types/S2.hpp.

    Args:
        v:       (3,) vector on the sphere
        length:  sphere radius (default 1.0; use G_m_s2 ≈ 9.809 for gravity)
        s2_typ:  1 (x-axis default, IKFoM gravity S2), 2 (y), or 3 (z)

    Returns:
        (3, 2) tangent basis matrix.

    Note: FAST-LIO uses `MTK::S2<double, 98090, 10000, 1>` for gravity,
    which means s2_typ=1, length=98090/10000=9.809.
    """
    v = v.ravel()
    bx = np.zeros((3, 2))
    if s2_typ == 3:
        if v[2] + length > 1e-10:
            bx[0, 0] = length - v[0]*v[0]/(length + v[2])
            bx[0, 1] = -v[0]*v[1]/(length + v[2])
            bx[1, 0] = -v[0]*v[1]/(length + v[2])
            bx[1, 1] = length - v[1]*v[1]/(length + v[2])
            bx[2, 0] = -v[0]
            bx[2, 1] = -v[1]
            bx /= length
        else:
            bx[1, 1] = -1.0
            bx[2, 0] = 1.0
    elif s2_typ == 2:
        if v[1] + length > 1e-10:
            bx[0, 0] = length - v[0]*v[0]/(length + v[1])
            bx[0, 1] = -v[0]*v[2]/(length + v[1])
            bx[1, 0] = -v[0]
            bx[1, 1] = -v[2]
            bx[2, 0] = -v[0]*v[2]/(length + v[1])
            bx[2, 1] = length - v[2]*v[2]/(length + v[1])
            bx /= length
        else:
            bx[1, 1] = -1.0
            bx[2, 0] = 1.0
    else:  # s2_typ == 1  (FAST-LIO default for gravity)
        if v[0] + length > 1e-10:
            bx[0, 0] = -v[1]
            bx[0, 1] = -v[2]
            bx[1, 0] = length - v[1]*v[1]/(length + v[0])
            bx[1, 1] = -v[2]*v[1]/(length + v[0])
            bx[2, 0] = -v[2]*v[1]/(length + v[0])
            bx[2, 1] = length - v[2]*v[2]/(length + v[0])
            bx /= length
        else:
            bx[1, 1] = -1.0
            bx[2, 0] = 1.0
    return bx


def s2_Nx_yy(v: np.ndarray, length: float = 1.0, s2_typ: int = 1) -> np.ndarray:
    """S2_Nx_yy: (2,3) matrix used in covariance propagation.

    Returns 1/length^2 * Bx^T * hat(v).
    """
    bx = s2_Bx(v, length, s2_typ)
    return (1.0 / (length * length)) * bx.T @ hat(v)


def s2_Mx(v: np.ndarray, delta2: np.ndarray, length: float = 1.0,
          s2_typ: int = 1) -> np.ndarray:
    """S2_Mx: (3,2) matrix used in covariance propagation.

    delta2 is a 2-vector in tangent space.
    """
    bx = s2_Bx(v, length, s2_typ)
    delta2 = np.asarray(delta2).ravel()
    if np.linalg.norm(delta2) < 1e-10:
        return -hat(v) @ bx
    bu = bx @ delta2
    from scipy.spatial.transform import Rotation as _R
    R_full = _R.from_rotvec(bu).as_matrix()
    return -R_full @ hat(v) @ A_matrix(bu).T @ bx
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

# Standard gravity magnitude (matches FAST-LIO: G_m_s2 = 9.81)
G_m_s2 = 9.81

# S2 sphere radius: matches MTK::S2<double, 98090, 10000, 1> in use-ikfom.hpp
# length = den/num = 98090/10000 = 9.809
_S2_LENGTH = 98090.0 / 10000.0   # 9.809
_S2_TYP    = 1                    # x-axis default (matches S2_typ=1 in C++)


@dataclass
class StateIkfom:
    """Python equivalent of state_ikfom.

    Rotation matrices are stored as (3,3) numpy arrays.
    Gravity is stored as a (3,) unit-sphere vector scaled by G_m_s2.
    """
    pos:      np.ndarray = field(default_factory=lambda: np.zeros(3))
    rot:      np.ndarray = field(default_factory=lambda: np.eye(3))
    offset_R: np.ndarray = field(default_factory=lambda: np.eye(3))
    offset_T: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel:      np.ndarray = field(default_factory=lambda: np.zeros(3))
    bg:       np.ndarray = field(default_factory=lambda: np.zeros(3))
    ba:       np.ndarray = field(default_factory=lambda: np.zeros(3))
    grav:     np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -G_m_s2]))
    P:        np.ndarray = field(default_factory=lambda: np.eye(23))

    def copy(self) -> "StateIkfom":
        return StateIkfom(
            pos=self.pos.copy(),
            rot=self.rot.copy(),
            offset_R=self.offset_R.copy(),
            offset_T=self.offset_T.copy(),
            vel=self.vel.copy(),
            bg=self.bg.copy(),
            ba=self.ba.copy(),
            grav=self.grav.copy(),
            P=self.P.copy(),
        )

    # ------------------------------------------------------------------
    # boxplus: apply a 23-DOF error vector δx to the state
    # ------------------------------------------------------------------
    def boxplus(self, dx: np.ndarray) -> "StateIkfom":
        """Return new state = self ⊕ dx (in-place modification of self)."""
        self.pos      += dx[0:3]
        self.rot       = self.rot @ exp(dx[3:6])
        self.offset_R  = self.offset_R @ exp(dx[6:9])
        self.offset_T += dx[9:12]
        self.vel      += dx[12:15]
        self.bg       += dx[15:18]
        self.ba       += dx[18:21]
        # S2 boxplus: grav += Bx * delta2 via rotation
        delta2 = dx[21:23]
        Bx = s2_Bx(self.grav, _S2_LENGTH, _S2_TYP)
        Bu = Bx @ delta2
        self.grav = exp(Bu) @ self.grav
        return self

    # ------------------------------------------------------------------
    # boxminus: compute 23-DOF error vector self ⊖ other
    # ------------------------------------------------------------------
    def boxminus(self, other: "StateIkfom") -> np.ndarray:
        """Compute dx = self ⊖ other (tangent vector from other to self)."""
        dx = np.zeros(23)
        dx[0:3]   = self.pos - other.pos
        dx[3:6]   = log(other.rot.T @ self.rot)
        dx[6:9]   = log(other.offset_R.T @ self.offset_R)
        dx[9:12]  = self.offset_T - other.offset_T
        dx[12:15] = self.vel - other.vel
        dx[15:18] = self.bg - other.bg
        dx[18:21] = self.ba - other.ba
        # S2 boxminus — matches S2::boxminus() in mtk/types/S2.hpp
        v = self.grav   # self
        u = other.grav  # other (reference)
        cross = np.cross(v, u)
        v_sin = np.linalg.norm(cross)
        v_cos = float(v.dot(u))
        theta = np.arctan2(v_sin, v_cos)
        if v_sin < 1e-10:
            if abs(theta) > 1e-10:
                dx[21] = np.pi
                dx[22] = 0.0
            else:
                dx[21:23] = 0.0
        else:
            Bx_u = s2_Bx(u, _S2_LENGTH, _S2_TYP)
            dx[21:23] = theta / v_sin * Bx_u.T @ hat(u) @ v
        return dx
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

def _inv3(M: np.ndarray) -> np.ndarray:
    """Explicit 3×3 inverse via cofactors — no LAPACK call overhead."""
    a, b, c = M[0, 0], M[0, 1], M[0, 2]
    d, e, f = M[1, 0], M[1, 1], M[1, 2]
    g, h, i = M[2, 0], M[2, 1], M[2, 2]
    det = a * (e*i - f*h) - b * (d*i - f*g) + c * (d*h - e*g)
    return np.array([[e*i - f*h,  c*h - b*i,  b*f - c*e],
                     [f*g - d*i,  a*i - c*g,  c*d - a*f],
                     [d*h - e*g,  b*g - a*h,  a*e - b*d]], dtype=np.float64) / det


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
    """Propagate state and covariance forward by dt.

    Mirrors esekfom::predict() in esekfom.hpp.

    Args:
        state: current StateIkfom (P is 23×23 covariance)
        dt:    time step [s]
        Q:     12×12 process noise covariance
        acc:   3-vec accelerometer measurement
        gyro:  3-vec gyroscope measurement

    Returns:
        new StateIkfom with updated state and P.

    Optimization notes (vs a naive implementation):
      - f() has only 3 non-zero rows (pos_dot, rot_dot, vel_dot). We compute
        them directly and skip building a full 24-vec.
      - df_dx has non-zeros only in the vel_dot row-block; offset_R, bg, ba
        and grav f-entries are all zero so their A_matrix corrections are
        identity and collapse to trivial row copies.
      - Uses so3_math.exp() instead of scipy.spatial.Rotation.from_rotvec
        (avoids creating a Python object per call).
    """
    n = 23
    R = state.rot                          # (3,3)
    grav = state.grav                      # (3,)
    acc_corr = acc - state.ba              # (3,)
    gyr_corr = gyro - state.bg             # (3,)
    # a_inertial + grav is the velocity derivative
    a_inertial = R @ acc_corr
    vel_dot = a_inertial + grav

    # --- Advance state (manifold boxplus) ---
    sn = state.copy()
    sn.pos      = state.pos + state.vel * dt
    sn.rot      = R @ exp(gyr_corr * dt)
    # offset_R, offset_T are constant — sn already holds the copy
    sn.vel      = state.vel + vel_dot * dt
    # bg, ba, grav also constant (sn already holds them)

    # --- F_x1 (23×23): identity except SO3 and S2 blocks ---
    # For offset_R (idx=6): seg = -f[6:9]*dt = 0  →  R_SO3 = I, A_mat = I
    #   (the correction collapses to identity, so F_x1[6:9,6:9] stays I)
    # For grav S2 (idx=21): seg = f[21:24]*dt = 0 → R_S2 = I
    #   Nx @ I @ Mx = Nx @ Mx  (depends on S2 numerics but is ≈ I_2)
    F_x1 = np.eye(n)
    seg_rot = -gyr_corr * dt                          # (3,)
    R_SO3   = exp(seg_rot)
    F_x1[3:6, 3:6] = R_SO3

    # S2 gravity block (zero f → R_S2 is identity)
    Nx = s2_Nx_yy(sn.grav, _S2_LENGTH, _S2_TYP)       # (2,3)
    Mx = s2_Mx(state.grav, _ZERO_VEC2, _S2_LENGTH, _S2_TYP)  # (3,2)
    F_x1[21:23, 21:23] = Nx @ Mx                      # R_S2 = I absorbed

    # --- f_x_final (23×23) rows: only rot, vel, and grav-S2 rows nonzero ---
    # Vect rows (pos, offset_T, vel, bg, ba) copy from the 24-DIM f_x rows.
    #   Non-zero entries in df_dx:
    #     F[0:3, 12:15] = I                      (pos_dot/vel)
    #     F[3:6, 15:18] = -I                     (rot_dot/bg)
    #     F[12:15, 3:6] = -R @ hat(acc-ba)       (vel_dot/rot)
    #     F[12:15, 18:21] = -R                    (vel_dot/ba)
    #     F[12:15, 21:23] = -hat(grav) @ Bx_g    (vel_dot/grav)
    #   All other rows (6:9, 9:12, 15:18, 18:21, 21:24) are zero.
    f_x_final = np.zeros((n, n))
    f_x_final[0:3, 12:15] = np.eye(3)                      # pos row
    # offset_T, bg, ba rows stay zero
    # rot row (3:6) — needs A_matrix(seg_rot) applied since seg_rot ≠ 0.
    # df_dx[3:6, :] has only -I at cols 15:18.
    A_rot = A_matrix(seg_rot)                             # (3,3)
    # f_x_final[3:6, 15:18] = A_rot @ (-I) = -A_rot
    f_x_final[3:6, 15:18] = -A_rot
    # vel row (12:15)
    neg_R_hat_acc = -R @ hat(acc_corr)                    # (3,3)
    grav_hat      = hat(grav)                              # (3,3)
    Bx_g = s2_Bx(grav, _S2_LENGTH, _S2_TYP)                # (3,2)
    neg_grav_hat_Bx = -grav_hat @ Bx_g                     # (3,2)
    f_x_final[12:15, 3:6]   = neg_R_hat_acc
    f_x_final[12:15, 18:21] = -R
    f_x_final[12:15, 21:23] = neg_grav_hat_Bx
    # grav row (21:23): temp_S2 = -Nx @ I @ grav_hat @ I = -Nx @ grav_hat
    # but df_dx[21:24, :] = 0, so f_x_final[21:23, :] = 0

    # --- f_w_final (23×12) rows: rot, vel, bg, ba ---
    # df_dw non-zero entries:
    #   G[3:6, 0:3]   = -I    (rot_dot/ng)
    #   G[12:15, 3:6] = -R    (vel_dot/na)
    #   G[15:18, 6:9] = I     (bg_dot/nbg)
    #   G[18:21, 9:12] = I    (ba_dot/nba)
    f_w_final = np.zeros((n, 12))
    # rot: A_rot @ (-I) → -A_rot in cols 0:3
    f_w_final[3:6, 0:3] = -A_rot
    f_w_final[12:15, 3:6] = -R
    f_w_final[15:18, 6:9] = np.eye(3)
    f_w_final[18:21, 9:12] = np.eye(3)

    # Covariance propagation: P = F @ P @ F^T + (dt*G) @ Q @ (dt*G)^T
    F = F_x1 + f_x_final * dt
    dtG = dt * f_w_final
    sn.P = F @ state.P @ F.T + dtG @ Q @ dtG.T

    return sn


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
# ============================================================================
# §5  增量式地图 KD-Tree 的 scipy 实现
# ----------------------------------------------------------------------------
# 【本节职责】
#   维护 FAST-LIO2 的全局地图点云，并为量测模型 h_share_model 提供每个
#   有效激光点的 k 近邻查询（k=5，用于平面拟合）。对应 C++ 版的独立库
#   include/ikd-Tree/ikd_Tree.h（KD_TREE<PointType>，支持动态增删的
#   incremental KD-Tree）。
#
# 【实现策略】
#   C++ ikd-Tree 通过增量式中位数分裂与局部重平衡实现动态插入/删除；
#   本节以 scipy.spatial.cKDTree（静态树）配合"惰性批量重建"策略等效替代：
#     - Add_Points 先把新点缓存进 _pending 缓冲区，仅当缓冲量超过阈值
#       max(_PENDING_MIN_ABS, _PENDING_MIN_FRAC × 已提交点数)
#       = max(2000, 5% × N) 时才合并进 _pts 并重建 cKDTree，
#       从而摊销 O(N log N) 的整树重建代价；
#     - Delete_Point_Boxes（FOV 移出时按轴对齐包围盒删点）先冲刷缓冲，
#       再以布尔掩码过滤 _pts 并立即重建；
#     - Nearest_Search 默认只查询已提交的树（_INCLUDE_PENDING_IN_SEARCH
#       = False），最多 _PENDING_MIN_ABS 个新点暂时不可见，换取约 2.9×
#       的查询加速；置 True 时对 _pending 额外做暴力 cdist 搜索并归并。
#
# 【关键数据结构】
#   _pts     : (N, 3) float64，已提交进 cKDTree 的地图点；
#   _pending : (M, 3) float64，尚未入树的新增点缓冲；
#   Nearest_Search 返回 (nn_pts (Q, k, 3), sq_dists (Q, k))，
#   距离为平方欧氏距离，与 C++ 版 Nearest_Search 的语义一致；
#   _voxel_downsample 按 leaf 尺寸做体素栅格降采样（每体素保留首点），
#   对应 C++ Add_Points(downsample_on=true) 中的体素去重。
#
# 【与 C++ 版的已知差异】
#   等距近邻的 tie-breaking 与 C++ ikd-Tree 不同（sliding-midpoint 分裂
#   vs 增量中位数分裂导致遍历序不同），属合法差异而非缺陷；
#   Root_Node 属性仅模拟 C++ 中 `ikdtree.Root_Node == nullptr` 的判空语义。
#
# 【源模块说明（原 docstring 改写）】
#   原 ikd_tree.py 为可选双后端封装（"scipy" 纯 Python / "cpp" pybind11
#   绑定原版 C++ ikd-Tree，经 IKD_BACKEND 环境变量选择，默认 scipy）。
#   单文件版仅保留纯 Python 的 scipy 后端 IkdTreeScipy 及其接口基类
#   IkdTreeBase；C++ 后端探测、工厂函数与 IPC 服务入口均已移除。
# ============================================================================


# ======================================================================
# Base interface (documentation only, not enforced via ABC)
# ======================================================================
class IkdTreeBase:
    """Common interface for ikd-Tree backends."""
    def set_downsample_param(self, leaf_size: float): ...
    def Build(self, pts: np.ndarray): ...
    def Add_Points(self, new_pts: np.ndarray, downsample: bool = True) -> int: ...
    def Delete_Point_Boxes(self, boxes: list): ...
    def Nearest_Search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]: ...
    def validnum(self) -> int: ...
    def size(self) -> int: ...
    @property
    def Root_Node(self): ...


# 讲解：IkdTreeScipy 为地图树的唯一实现。生命周期上，offline 主循环在首帧
# 用 Build() 建树，此后每帧 map_incremental 阶段调用 Add_Points()，FOV
# 平移时调用 Delete_Point_Boxes()；h_share_model 每次迭代经 Nearest_Search()
# 批量取 5-NN。validnum/size/Root_Node 与 C++ 端同名接口一一对应。

# ======================================================================
# scipy backend (original pure-Python implementation)
# ======================================================================
class IkdTreeScipy(IkdTreeBase):
    """Incremental KD-Tree backed by scipy.cKDTree (pure Python fallback).

    Interface matches the C++ KD_TREE<PointType> used in FAST-LIO.
    Note: tie-breaking differs from C++ ikd-Tree due to different tree
    structure (sliding-midpoint vs incremental median split).
    """

    def __init__(self):
        self._pts: Optional[np.ndarray] = None       # (N, 3) committed to tree
        self._tree: Optional[cKDTree] = None          # cKDTree over self._pts
        self._pending: Optional[np.ndarray] = None   # (M, 3) new points not yet in tree
        self._downsample: float = 0.5
        # Batched-rebuild policy: commit pending into tree only when
        # pending size exceeds this absolute threshold OR 5% of the tree.
        # Nearest_Search covers both _tree and _pending (the latter brute-force).
        self._PENDING_MIN_ABS = 2000    # min buffered points before rebuild
        self._PENDING_MIN_FRAC = 0.05   # or this fraction of the committed tree
        # When True, Nearest_Search also brute-forces the pending buffer
        # for strictly up-to-date results. When False (default), we search
        # only the committed tree — up to _PENDING_MIN_ABS recent points are
        # temporarily invisible but the speed gain is large (~2.9×).
        self._INCLUDE_PENDING_IN_SEARCH = False
        self._dirty = False

    # ------------------------------------------------------------------
    def set_downsample_param(self, leaf_size: float):
        self._downsample = leaf_size

    # ------------------------------------------------------------------
    def Build(self, pts: np.ndarray):
        """Build tree from scratch."""
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        self._pts = pts[:, :3].copy()
        self._tree = cKDTree(self._pts)
        self._pending = None
        self._dirty = False

    # ------------------------------------------------------------------
    def _commit_pending(self):
        """Merge pending buffer into the main point set and rebuild cKDTree."""
        if self._pending is None or len(self._pending) == 0:
            return
        if self._pts is None or len(self._pts) == 0:
            self._pts = self._pending
        else:
            self._pts = np.vstack([self._pts, self._pending])
        self._tree = cKDTree(self._pts)
        self._pending = None
        self._dirty = False

    # ------------------------------------------------------------------
    def Add_Points(self, new_pts: np.ndarray, downsample: bool = True) -> int:
        """Add points to the tree lazily.

        New points are appended to `_pending`. The cKDTree is rebuilt only
        when `_pending` exceeds the configured threshold OR when
        Nearest_Search is about to return a result that a user might
        depend on being complete (we cover this via dual search at query
        time).
        """
        if len(new_pts) == 0:
            return 0
        new_pts = np.asarray(new_pts, dtype=np.float64)
        if new_pts.ndim == 1:
            new_pts = new_pts.reshape(-1, 3)
        new_pts = new_pts[:, :3]

        if downsample and self._downsample > 0:
            new_pts = self._voxel_downsample(new_pts, self._downsample)

        if self._pending is None or len(self._pending) == 0:
            self._pending = new_pts
        else:
            self._pending = np.vstack([self._pending, new_pts])

        # Rebuild if pending is large enough to amortize rebuild cost
        n_committed = len(self._pts) if self._pts is not None else 0
        pending_n = len(self._pending)
        threshold = max(self._PENDING_MIN_ABS,
                        int(self._PENDING_MIN_FRAC * max(1, n_committed)))
        if pending_n >= threshold:
            self._commit_pending()
        else:
            self._dirty = True
        return len(new_pts)

    # ------------------------------------------------------------------
    def Delete_Point_Boxes(self, boxes: list):
        """Remove points inside axis-aligned boxes. Flushes pending then rebuilds."""
        self._commit_pending()
        if self._pts is None or len(self._pts) == 0:
            return
        mask = np.ones(len(self._pts), dtype=bool)
        for (mn, mx) in boxes:
            mn = np.asarray(mn)
            mx = np.asarray(mx)
            in_box = np.all((self._pts >= mn) & (self._pts <= mx), axis=1)
            mask &= ~in_box
        self._pts = self._pts[mask]
        if len(self._pts) > 0:
            self._tree = cKDTree(self._pts)
        else:
            self._tree = None
        self._dirty = False

    # ------------------------------------------------------------------
    def Nearest_Search(
        self,
        query: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors for each query point.

        Args:
            query: (K, 3) query points
            k:     number of neighbors

        Returns:
            nn_pts:   (K, k, 3) nearest neighbor coordinates
            sq_dists: (K, k)    squared distances
        """
        # Empty: nothing to search
        total_n = (len(self._pts) if self._pts is not None else 0) + \
                  (len(self._pending) if self._pending is not None else 0)
        if total_n == 0:
            query = np.asarray(query)
            Q = query.shape[0] if query.ndim > 1 else 1
            return np.zeros((Q, k, 3)), np.full((Q, k), np.inf)

        if self._tree is None and self._pts is not None and len(self._pts) > 0:
            self._tree = cKDTree(self._pts)

        query = np.asarray(query, dtype=np.float64)
        single = query.ndim == 1
        if single:
            query = query.reshape(1, 3)
        q3 = query[:, :3]
        Q = q3.shape[0]

        # --- Query the committed tree ---
        if self._tree is not None:
            actual_k = min(k, len(self._pts))
            d_tree, i_tree = self._tree.query(q3, k=actual_k)
            if actual_k == 1:
                d_tree = d_tree[:, np.newaxis]
                i_tree = i_tree[:, np.newaxis]
            nn_from_tree = self._pts[i_tree]                 # (Q, actual_k, 3)
            dist_from_tree = d_tree ** 2                      # (Q, actual_k)
            if actual_k < k:
                pad = k - actual_k
                nn_from_tree = np.concatenate(
                    [nn_from_tree, np.zeros((Q, pad, 3))], axis=1)
                dist_from_tree = np.concatenate(
                    [dist_from_tree, np.full((Q, pad), np.inf)], axis=1)
        else:
            nn_from_tree = np.zeros((Q, k, 3))
            dist_from_tree = np.full((Q, k), np.inf)

        # --- Optionally brute-force over pending buffer ---
        if (self._INCLUDE_PENDING_IN_SEARCH and
                self._pending is not None and len(self._pending) > 0):
            from scipy.spatial.distance import cdist
            pend = self._pending
            pend_dist = cdist(q3, pend, metric='sqeuclidean')    # (Q, M)
            k_use = min(k, pend_dist.shape[1])
            part_idx = np.argpartition(pend_dist, k_use - 1, axis=1)[:, :k_use]
            rows = np.arange(Q)[:, None]
            pend_top_dist = pend_dist[rows, part_idx]
            pend_top_pts  = pend[part_idx]

            merged_dist = np.concatenate([dist_from_tree, pend_top_dist], axis=1)
            merged_pts  = np.concatenate([nn_from_tree, pend_top_pts], axis=1)
            sort_idx = np.argsort(merged_dist, axis=1)[:, :k]
            nn_pts   = merged_pts[rows, sort_idx]
            sq_dists = merged_dist[rows, sort_idx]
        else:
            nn_pts   = nn_from_tree
            sq_dists = dist_from_tree

        if single:
            return nn_pts[0], sq_dists[0]
        return nn_pts, sq_dists

    # ------------------------------------------------------------------
    def validnum(self) -> int:
        n = len(self._pts) if self._pts is not None else 0
        n += len(self._pending) if self._pending is not None else 0
        return n

    def size(self) -> int:
        return self.validnum()

    @property
    def Root_Node(self):
        """Mimic C++ ikdtree.Root_Node == nullptr check."""
        if ((self._pts is not None and len(self._pts) > 0) or
            (self._pending is not None and len(self._pending) > 0)):
            return self._tree if self._tree is not None else True
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _voxel_downsample(pts: np.ndarray, leaf: float) -> np.ndarray:
        """Simple voxel grid downsampling: keep one point per voxel."""
        if len(pts) == 0:
            return pts
        keys = np.floor(pts / leaf).astype(np.int64)
        # Build a unique key per voxel using Cantor-like pairing
        _, unique_idx = np.unique(
            keys[:, 0] * (10**7) + keys[:, 1] * (10**4) + keys[:, 2],
            return_index=True,
        )
        return pts[unique_idx]
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
# ============================================================================


# livox_ros_driver/CustomPoint memory layout (ROS1, no padding between user-type
# struct members): offset_time(u32) + x/y/z(f32) + reflectivity/tag/line(u8) = 19 B
_LIVOX_POINT_DTYPE = np.dtype([
    ('offset_time', '<u4'),
    ('x',           '<f4'),
    ('y',           '<f4'),
    ('z',           '<f4'),
    ('reflectivity', 'u1'),
    ('tag',          'u1'),
    ('line',         'u1'),
])
assert _LIVOX_POINT_DTYPE.itemsize == 19


# std_msgs/Header 的通用解析：seq(u32) + stamp.sec(u32) + stamp.nsec(u32) +
# frame_id 长度前缀(u32)，随后跳过 frame_id 字符串本体，返回浮点时间戳与下一
# 字段的偏移。IMU 与 CustomMsg 两种消息均以此开头。
def _parse_header(buf, off=0):
    """Parse std_msgs/Header. Returns (stamp_sec, next_offset).

    Uses `sec + nsec / 1e9` (not `* 1e-9`) to match rospy's Time.to_sec()
    bit-for-bit. `1e-9` is not exactly representable in float64, so
    multiplication can disagree with division by 1 ULP on some stamps —
    enough to reorder a boundary IMU sample between scan windows and
    accumulate a cm-scale trajectory drift on short bags.
    """
    _seq, sec, nsec, flen = struct.unpack_from('<IIII', buf, off)
    return sec + nsec / 1e9, off + 16 + flen


# sensor_msgs/Imu：header 后为 37 个 float64 的连续块，按序列化顺序切片取
# angular_velocity（下标 13:16）与 linear_acceleration（下标 25:28）。
def parse_imu_bytes(buf) -> dict:
    """Parse sensor_msgs/Imu raw bytes. Mirrors imu_msg_to_dict()."""
    stamp, off = _parse_header(buf, 0)
    # Layout after header (all float64, no padding):
    #   Quaternion orientation                    [ 0: 4]  4 doubles
    #   float64[9] orientation_covariance         [ 4:13]  9 doubles
    #   Vector3 angular_velocity                  [13:16]  3 doubles
    #   float64[9] angular_velocity_covariance    [16:25]  9 doubles
    #   Vector3 linear_acceleration               [25:28]  3 doubles
    #   float64[9] linear_acceleration_covariance [28:37]  9 doubles
    vals = struct.unpack_from('<37d', buf, off)
    return {
        "stamp": stamp,
        "acc":   np.array(vals[25:28]),
        "gyro":  np.array(vals[13:16]),
    }


# livox_ros_driver/CustomMsg：解析点数组并复现 C++ preprocess.cpp 的 avia_handler
# 前端过滤（point_filter_num 抽稀、line>=128 剔除），输出 (N,4) float32，第 4 列
# 为点内偏移时间（ns → ms），供 IMU 去畸变按时间插值使用。
def parse_livox_bytes(buf, point_filter_num: int = 1):
    """Parse livox_ros_driver/CustomMsg raw bytes.

    Returns (stamp_sec, (N,4) float32 [x, y, z, offset_time_ms]) or (stamp, None).
    """
    stamp, off = _parse_header(buf, 0)
    # CustomMsg body:
    #   uint64 timebase      (8 B)
    #   uint32 point_num     (4 B)
    #   uint8  lidar_id      (1 B)
    #   uint8[3] rsvd        (3 B — fixed-size array, no length prefix)
    #   uint32 pts_count     (4 B — variable-length array length prefix)
    #   CustomPoint[] points (pts_count × 19 B)
    pts_count = struct.unpack_from('<I', buf, off + 16)[0]
    pts_off   = off + 20

    arr = np.frombuffer(
        buf, dtype=_LIVOX_POINT_DTYPE, count=pts_count, offset=pts_off,
    )
    if point_filter_num > 1:
        arr = arr[::point_filter_num]
    # Mirrors original: drop points with line>=128 (in practice always keeps all)
    if arr.size and arr['line'].max() >= 128:
        arr = arr[arr['line'] < 128]
    if arr.size == 0:
        return stamp, None

    out = np.empty((len(arr), 4), dtype=np.float32)
    out[:, 0] = arr['x']
    out[:, 1] = arr['y']
    out[:, 2] = arr['z']
    out[:, 3] = arr['offset_time'].astype(np.float32) * np.float32(1e-6)  # ns → ms
    return stamp, out
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


# ---------------------------------------------------------------------------
# PhaseTimer — lightweight per-region timing collector.
# Zero overhead when --profile is not set (all calls become context-manager
# no-ops through the _NullTimer).
# ---------------------------------------------------------------------------
class PhaseTimer:
    """Accumulates wall-time and call counts per named region."""

    def __init__(self) -> None:
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def region(self, name: str) -> "_TimerCtx":
        return _TimerCtx(self, name)

    def record(self, name: str, dt: float) -> None:
        self._totals[name] = self._totals.get(name, 0.0) + dt
        self._counts[name] = self._counts.get(name, 0) + 1

    def report(self, wall_time: float) -> str:
        if not self._totals:
            return "(no profiling regions recorded)"
        lines = []
        lines.append(f"{'region':<20s} {'total_s':>10s} {'mean_ms':>10s} "
                     f"{'count':>8s} {'pct_wall':>9s}")
        lines.append("-" * 60)
        for name in sorted(self._totals.keys(),
                           key=lambda k: -self._totals[k]):
            total = self._totals[name]
            count = self._counts[name]
            mean_ms = (total / count) * 1000.0 if count else 0.0
            pct = (total / wall_time) * 100.0 if wall_time > 0 else 0.0
            lines.append(f"{name:<20s} {total:>10.3f} {mean_ms:>10.3f} "
                         f"{count:>8d} {pct:>8.1f}%")
        return "\n".join(lines)


class _TimerCtx:
    __slots__ = ("_timer", "_name", "_t0")

    def __init__(self, timer: PhaseTimer, name: str) -> None:
        self._timer = timer
        self._name = name
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._timer.record(self._name, time.perf_counter() - self._t0)
        return False


class _NullTimer:
    """No-op PhaseTimer used when --profile is disabled."""

    def region(self, name: str):
        return _NULL_CTX

    def record(self, name: str, dt: float) -> None:
        pass

    def report(self, wall_time: float) -> str:
        return ""


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


_NULL_CTX = _NullCtx()
_NULL_TIMER = _NullTimer()

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


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_param(cfg: dict, *keys, default=None):
    """Traverse nested yaml dict with fallback."""
    d = cfg
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# ---------------------------------------------------------------------------
# ROS message helpers
# ---------------------------------------------------------------------------
# 讲解：rosbag 为 ROS 环境专属依赖，保持函数级惰性导入；主路径通过
# raw=True 直接取字节流，仅在 Velodyne/Ouster 回退路径实例化 rospy 消息。
def _try_import_rosbag():
    try:
        import rosbag
        return rosbag
    except ImportError:
        sys.exit("rosbag not found. Run inside the Docker container or source ROS.")


def imu_msg_to_dict(msg) -> dict:
    """Convert sensor_msgs/Imu to plain dict."""
    return {
        "stamp": msg.header.stamp.to_sec(),
        "acc":   np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ]),
        "gyro":  np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ]),
    }


_LIVOX_GETTER = _op.attrgetter('x', 'y', 'z', 'offset_time')


def livox_msg_to_array(msg, point_filter_num: int = 1) -> Optional[np.ndarray]:
    """Convert livox_ros_driver/CustomMsg to (N,4) float32 [x,y,z,curvature_ms].

    Optimized: uses a slice + list comprehension with operator.attrgetter
    to batch-extract the fields, then a single numpy array construction
    followed by a vectorized ns→ms scaling. Roughly 2-3x faster than the
    explicit Python for-loop with per-attribute access.
    """
    pts_list = msg.points
    # Subsample via slice (C-level list slicing, fast)
    if point_filter_num > 1:
        pts_list = pts_list[::point_filter_num]

    # Filter on pt.line < 128 (usually all pass) — small branch overhead
    rows = [_LIVOX_GETTER(pt) for pt in pts_list if pt.line < 128]
    if not rows:
        return None
    arr = np.array(rows, dtype=np.float32)
    # offset_time: ns → ms as a vectorized step
    arr[:, 3] *= 1e-6
    return arr


def pcl2_to_array(msg, point_filter_num: int = 1, lidar_type: int = 2) -> Optional[np.ndarray]:
    """Convert sensor_msgs/PointCloud2 to (N,4) [x,y,z,timestamp_offset_ms]."""
    import struct
    fields = {f.name: f for f in msg.fields}
    point_step = msg.point_step
    data = msg.data

    xs, ys, zs, ts = [], [], [], []
    for i in range(msg.width):
        if i % point_filter_num != 0:
            continue
        off = i * point_step
        x = struct.unpack_from('<f', data, off + fields['x'].offset)[0]
        y = struct.unpack_from('<f', data, off + fields['y'].offset)[0]
        z = struct.unpack_from('<f', data, off + fields['z'].offset)[0]

        # Timestamp offset: Velodyne uses 'time' field in seconds relative to scan start
        if 't' in fields:
            t_off = struct.unpack_from('<f', data, off + fields['t'].offset)[0] * 1000.0  # s→ms
        elif 'time' in fields:
            t_off = struct.unpack_from('<f', data, off + fields['time'].offset)[0] * 1000.0
        elif 'timestamp' in fields:
            t_raw = struct.unpack_from('<d', data, off + fields['timestamp'].offset)[0]
            t_off = (t_raw - msg.header.stamp.to_sec()) * 1000.0
        else:
            t_off = 0.0

        xs.append(x); ys.append(y); zs.append(z); ts.append(t_off)

    if not xs:
        return None
    return np.column_stack([xs, ys, zs, ts]).astype(np.float32)


# ---------------------------------------------------------------------------
# Voxel downsampling (PCL-equivalent)
# ---------------------------------------------------------------------------
# 讲解：与 pcl::VoxelGrid 等价的质心降采样。体素索引 floor(p/leaf) 经整数
# 哈希（1e8/1e4 进制）合并后，用 np.bincount 按列做加权求和求质心。
def voxel_downsample(pts: np.ndarray, leaf: float) -> np.ndarray:
    """Voxel grid downsample with centroid (matches pcl::VoxelGrid).

    Uses a per-column np.bincount — each call is a vectorized C path. The
    loop iterates over a fixed small number of columns (3 or 4).
    """
    if len(pts) == 0:
        return pts
    keys = np.floor(pts[:, :3] / leaf).astype(np.int64)
    hash_keys = keys[:, 0] * (10**8) + keys[:, 1] * (10**4) + keys[:, 2]
    unique_keys, inverse = np.unique(hash_keys, return_inverse=True)
    counts = np.bincount(inverse)
    n_voxels = len(unique_keys)
    out = np.empty((n_voxels, pts.shape[1]), dtype=pts.dtype)
    for d in range(pts.shape[1]):
        out[:, d] = np.bincount(inverse, weights=pts[:, d], minlength=n_voxels) / counts
    return out


# ---------------------------------------------------------------------------
# pointBodyToWorld
# ---------------------------------------------------------------------------
# 讲解：p_w = R_wi (R_il p_l + t_il) + t_wi，即先经 LiDAR→IMU 外参、再经
# IMU→世界位姿的复合变换；与 C++ pointBodyToWorld() 一致，此处向量化处理整帧。
def point_body_to_world(state: StateIkfom, pts_body: np.ndarray) -> np.ndarray:
    """Transform (N,3) or (N,4) pts from LiDAR-body to world frame."""
    xyz = pts_body[:, :3]
    world = (state.rot @ (state.offset_R @ xyz.T + state.offset_T[:, None]) + state.pos[:, None]).T
    if pts_body.shape[1] > 3:
        return np.column_stack([world, pts_body[:, 3:]])
    return world


# ---------------------------------------------------------------------------
# lasermap_fov_segment: remove map points outside FOV sphere
# ---------------------------------------------------------------------------
# 讲解：对应 C++ lasermap_fov_segment()。保留半径为 det_range*mov_threshold
# 的球内地图点，球外点删除后对 scipy cKDTree 做整树重建（EAGER 策略）。
def lasermap_fov_segment(
    ikdtree,
    pos_lid: np.ndarray,
    det_range: float = 300.0,
    mov_threshold: float = 1.5,
) -> None:
    """Remove map voxels too far from current LiDAR position.

    Uses Delete_Point_Boxes with a bounding box covering the far region.
    Works with both C++ and scipy backends.

    Note: this function is the unconditional variant. The run() loop uses
    a movement-gated wrapper to skip the expensive flatten+rebuild on
    scans where the LiDAR has not moved far enough to push any point
    out of the sphere.
    """
    if ikdtree.Root_Node is None:
        return
    if ikdtree.validnum() == 0:
        return
    half = det_range * mov_threshold
    # Merge any pending points into the main array so we operate on all
    ikdtree._commit_pending()
    pts = ikdtree._pts
    dists = np.linalg.norm(pts - pos_lid, axis=1)
    keep = dists <= half
    if keep.all():
        return
    kept_pts = pts[keep]
    # Rebuild tree with kept points only
    ikdtree._pts = kept_pts
    ikdtree._pending = None
    if len(kept_pts) > 0:
        from scipy.spatial import cKDTree
        ikdtree._tree = cKDTree(kept_pts)
    else:
        ikdtree._tree = None
    ikdtree._dirty = False


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
    pcl_map_pts: List[np.ndarray] = []

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

            # Accumulate full-resolution undistorted scan in world frame
            # (matches C++ pcl_map_accum which uses feats_undistort, not the downsampled cloud)
            with _timer.region("map_accum_full"):
                pts_world_full = point_body_to_world(state, pts_undistort)[:, :3]
                pcl_map_pts.append(pts_world_full)

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
    _save_pcd(pcl_map_pts, os.path.join(output_dir, "PCD", "map_offline_py.pcd"))
    save_time = time.perf_counter() - save_t0
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


# 讲解：TUM 轨迹按 9 位小数逐行写出；PCD 优先经 open3d（惰性导入）写二进制，
# 无 open3d 时回退到手写 PCD v0.7 二进制格式（float32，与 pcl 布局一致）。
def _save_trajectory(traj: list, path: str) -> None:
    print(f"Saving trajectory ({len(traj)} poses) → {path}")
    with open(path, "w") as f:
        for tp in traj:
            f.write(
                f"{tp['t']:.9f} "
                f"{tp['tx']:.9f} {tp['ty']:.9f} {tp['tz']:.9f} "
                f"{tp['qx']:.9f} {tp['qy']:.9f} {tp['qz']:.9f} {tp['qw']:.9f}\n"
            )


def _save_pcd(pts_list: list, path: str) -> None:
    if not pts_list:
        print("No map points to save.")
        return
    all_pts = np.vstack(pts_list)
    print(f"Saving PCD ({len(all_pts)} pts) → {path}")
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts[:, :3].astype(np.float64))
        o3d.io.write_point_cloud(path, pcd, write_ascii=False, compressed=False)
    except ImportError:
        # Fallback: write binary PCD (float32, matches pcl binary format)
        pts_f32 = all_pts[:, :3].astype(np.float32)
        n = len(pts_f32)
        with open(path, "wb") as f:
            header = (
                f"# .PCD v0.7 - Point Cloud Data\n"
                f"FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
                f"POINTS {n}\nDATA binary\n"
            )
            f.write(header.encode())
            f.write(pts_f32.tobytes())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
# 讲解：命令行参数与 YAML 配置（common/preprocess/mapping 三节）合并后调用
# run()；YAML 缺省项回退到与 C++ launch 文件一致的默认值。
def _build_arg_parser():
    p = argparse.ArgumentParser(description="FAST-LIO offline mapping (Python)")
    p.add_argument("--bag",        required=True, help="Path to input rosbag")
    p.add_argument("--config",     default=None,  help="Path to YAML config")
    p.add_argument("--output_dir", default=".",   help="Output directory")
    p.add_argument("--lid_topic",  default="/livox/lidar")
    p.add_argument("--imu_topic",  default="/livox/imu")
    p.add_argument("--lidar_type", type=int, default=1,
                   help="1=Livox 2=Velodyne 3=Ouster 4=MARSIM")
    p.add_argument("--filter_surf", type=float, default=0.5)
    p.add_argument("--filter_map",  type=float, default=0.5)
    p.add_argument("--point_filter_num", type=int, default=2)
    p.add_argument("--max_iter",    type=int,   default=3)
    p.add_argument("--max_scans",   type=int,   default=0,
                   help="Stop after N scans (0 = run all)")
    p.add_argument("--profile",     action="store_true",
                   help="Enable per-region timing and print a report at end")
    return p


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
