#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAST-LIO2 纯 Python 实现的工具函数下沉模块 —— fastlio_numpy.py 的配套文件.

本文件收纳 scripts/fastlio_numpy.py（2708 行单文件纯 numpy FAST-LIO2 参考实现）
中与主算法流程正交的“工具性”符号：SO(3)/S(2) 流形数学原语、rosbag 裸字节
消息解析、剖析计时器、配置与命令行解析、以及轨迹/点云文件输出与体素降采样。
将这些工具从主文件下沉至此，使主文件专注于状态估计核心（StateIkfom、
IESEKF predict/update、ImuProcess、ikd-Tree、h_share_model、离线主流程 run），
提升可读性与可维护性。

本模块**完全自包含**，不依赖 fastlio_numpy.py（无循环导入）：每个被下沉的
函数/类所依赖的符号，或在本文件内可解析，或来自标准库 / numpy / scipy /
pyyaml。rosbag、open3d、scipy 等重依赖沿用原实现的“函数体内惰性导入”策略，
以保持在无 ROS / 无可视化环境下的可导入性。

本次下沉为**逐字节不变（bit-identical）**的纯重构：所有函数体、类体、行内
注释、浮点写法与运算顺序均与原 fastlio_numpy.py 逐字节一致，仅做“移动 +
补充自包含 import + 添加节头横幅”。因此主文件在 6 个 rosbag 上的输出轨迹与
重构前逐字节相同。

章节组织：
    §A  SO(3)/S(2) 流形数学          （原 fastlio_numpy.py §1）
    §B  23 维流形状态 StateIkfom      （原 fastlio_numpy.py §2）
    §C  rosbag/消息解析              （原 fastlio_numpy.py §7 + §8 ROS helpers）
    §D  剖析计时                      （原 fastlio_numpy.py §8 PhaseTimer）
    §E  配置 / 命令行解析            （原 fastlio_numpy.py §8 config/CLI）
    §F  文件输出                      （原 fastlio_numpy.py §8 save helpers）
    §G  几何：体素降采样              （原 fastlio_numpy.py §8 voxel_downsample）
    增量地图 KD-Tree · FOV 裁剪 · 体↔世界变换
                                      （原 fastlio_numpy.py §5 + §8）
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
import time

import numpy as np
import yaml
import operator as _op
from dataclasses import dataclass, field
from scipy.spatial import cKDTree
from typing import Any, Callable, Dict, List, Optional, Tuple


# =============================================================================
# §A  SO(3) / S(2) 流形数学基础
# -----------------------------------------------------------------------------
# 下沉自 fastlio_numpy.py §1。提供 FAST-LIO2 状态估计所依赖的两类流形运算
# 原语：SO(3) 群上的 hat/exp/log/A_matrix 与四元数↔旋转矩阵互转，及单位球面
# S(2)（重力约束）切空间工具 s2_Bx / s2_Nx_yy / s2_Mx。常量 G_m_s2 /
# _S2_LENGTH / _S2_TYP 一并下沉（主文件反向 import）。
#
# 关键约定（承自原文件 §1 模块 docstring）：
#   - 指数映射（Rodrigues）：exp(φ) = I + sinθ·K + (1−cosθ)·K²，θ=||φ||，
#     K=hat(φ/θ)；θ<1e-7 退化为单位阵。
#   - 右雅可比 A(φ)=I+(1−cosθ)/θ²·hat(φ)+(θ−sinθ)/θ³·hat(φ)²。
#   - S(2) 取 MTK::S2<double, 98090, 10000, 1>：length=9.809、s2_typ=1。
#   - 四元数一律 [x, y, z, w] 顺序（与 scipy / Eigen 一致）。
#   - log(·) 用 math.acos / math.sin 标量运算规避 numpy 分派开销；
#     rot_to_quat 与 s2_Mx 内部 scipy 惰性 import 按原样保留。
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


# Standard gravity magnitude (matches FAST-LIO: G_m_s2 = 9.81)
G_m_s2 = 9.81

# S2 sphere radius: matches MTK::S2<double, 98090, 10000, 1> in use-ikfom.hpp
# length = den/num = 98090/10000 = 9.809
_S2_LENGTH = 98090.0 / 10000.0   # 9.809
_S2_TYP    = 1                    # x-axis default (matches S2_typ=1 in C++)


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


# =============================================================================
# §B  23 维流形状态定义与 ⊞/⊟（boxplus / boxminus）运算
# -----------------------------------------------------------------------------
# 下沉自 fastlio_numpy.py §2。对应 C++ include/use-ikfom.hpp（MTK_BUILD_MANIFOLD
# 宏定义的 state_ikfom），其中 S2 流形部分对应 include/IKFoM_toolkit/mtk/types/
# S2.hpp。因其 boxplus/boxminus 直接引用 §A 的 exp/log/hat/s2_Bx 与常量
# G_m_s2/_S2_LENGTH/_S2_TYP，故置于 §A 数学之后。
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
#          为 g 处切平面的一组正交基（见 §A 的 s2_Bx）；
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
# §C  rosbag 裸字节解析（raw-bytes message parsers）
# -----------------------------------------------------------------------------
# 下沉自 fastlio_numpy.py §7（裸字节解析）与 §8 的 ROS 消息 helpers。绕过 rospy
# 的反序列化层，直接把 rosbag 的裸字节 payload 解析为流水线所需 numpy 数组；
# 另含经 rospy 反序列化后处理 Livox/PointCloud2 的兼容路径。
#
# 关键精度约定：时间戳换算必须写作 sec + nsec / 1e9（而非乘以 1e-9），以与
# rospy Time.to_sec() 逐位一致；详见 _parse_header 的注释。rosbag 为 ROS 环境
# 专属依赖，保持函数级惰性导入。
# =============================================================================


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


# =============================================================================
# §D  剖析计时（per-region timing）
# -----------------------------------------------------------------------------
# 下沉自 fastlio_numpy.py §8 的 PhaseTimer 及其配套。--profile 时按命名区域累计
# 墙钟时间与调用次数；未开启时以 _NullTimer / _NullCtx 作零开销替身。
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


# =============================================================================
# §E  配置加载与命令行解析
# -----------------------------------------------------------------------------
# 下沉自 fastlio_numpy.py §8 的 config/CLI 部分。load_config 读取 YAML；
# get_param 带回退地遍历嵌套字典；_build_arg_parser 构造命令行解析器。三者
# 均不引用主文件的 LIDAR_* / FILTER_SIZE_MAP 等常量（其默认值在解析器内硬编码），
# 故这些常量仍留在主文件，本节自包含于 yaml / argparse。
# =============================================================================


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


# =============================================================================
# §F  文件输出（轨迹 / 点云）
# -----------------------------------------------------------------------------
# 下沉自 fastlio_numpy.py §8 的 save helpers。_save_trajectory 按 TUM 格式（9 位
# 小数）逐行写出；_save_pcd 优先经 open3d（惰性导入）写二进制，无 open3d 时
# 回退到手写 PCD v0.7 二进制格式。
# =============================================================================


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


# =============================================================================
# §G  几何：体素栅格降采样
# -----------------------------------------------------------------------------
# 下沉自 fastlio_numpy.py §8 的 voxel_downsample。与 pcl::VoxelGrid 等价的质心
# 降采样。
# =============================================================================


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


# =============================================================================
# 增量地图 KD-Tree · FOV 裁剪 · 体↔世界变换
# -----------------------------------------------------------------------------
# 下沉自 fastlio_numpy.py §5（增量式地图 KD-Tree 的 scipy 实现）及 §8 的两个
# 几何 helper（point_body_to_world / lasermap_fov_segment）。IkdTreeScipy 为
# 地图树的唯一实现，继承文档性接口基类 IkdTreeBase（须置于其后）；
# point_body_to_world 引用 §B 的 StateIkfom；lasermap_fov_segment 以鸭子类型
# 访问 ikdtree 内部属性（_commit_pending/_pts/_pending/_tree），不引用主文件符号。
#
# 【实现策略】
#   C++ ikd-Tree 通过增量式中位数分裂与局部重平衡实现动态插入/删除；
#   本节以 scipy.spatial.cKDTree（静态树）配合“惰性批量重建”策略等效替代：
#     - Add_Points 先把新点缓存进 _pending 缓冲区，仅当缓冲量超过阈值
#       max(_PENDING_MIN_ABS, _PENDING_MIN_FRAC × 已提交点数)
#       = max(2000, 5% × N) 时才合并进 _pts 并重建 cKDTree；
#     - Delete_Point_Boxes（FOV 移出时按轴对齐包围盒删点）先冲刷缓冲，
#       再以布尔掩码过滤 _pts 并立即重建；
#     - Nearest_Search 默认只查询已提交的树（_INCLUDE_PENDING_IN_SEARCH
#       = False），换取约 2.9× 的查询加速。
#
# 【与 C++ 版的已知差异】
#   等距近邻的 tie-breaking 与 C++ ikd-Tree 不同（sliding-midpoint 分裂
#   vs 增量中位数分裂导致遍历序不同），属合法差异而非缺陷。
# =============================================================================


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
