"""
四元数和旋转相关的工具函数。
"""

import numpy as np


def quat_conj(q):
    """四元数共轭 (等价于单位四元数的逆)"""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(a, b):
    """Hamilton 四元数乘法 a * b，格式为 [w, x, y, z]"""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_to_mat(q):
    """单位四元数转 3x3 旋转矩阵"""
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-8)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_rotate_inv(q, v):
    """用四元数的逆旋转向量 (将世界系向量转换到局部坐标系)"""
    return quat_to_mat(q).T @ np.asarray(v, dtype=np.float64)


def heading_quat(q):
    """提取四元数的偏航(yaw)分量，返回仅含 yaw 的四元数"""
    direction = quat_to_mat(q) @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    heading = np.arctan2(direction[1], direction[0])
    return np.array([np.cos(heading / 2.0), 0.0, 0.0, np.sin(heading / 2.0)], dtype=np.float64)


def heading_quat_inv(q):
    """提取四元数偏航分量的逆"""
    direction = quat_to_mat(q) @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    heading = -np.arctan2(direction[1], direction[0])
    return np.array([np.cos(heading / 2.0), 0.0, 0.0, np.sin(heading / 2.0)], dtype=np.float64)


def rot6_from_quat_delta(base_quat_wxyz, ref_quat_wxyz):
    """
    计算两个四元数之间的相对旋转，并提取旋转矩阵的前两列 (6D 旋转表示)。
    6D 旋转表示比四元数更适合神经网络学习，避免了万向节锁和不连续性。
    """
    rel = quat_mul(quat_conj(base_quat_wxyz), ref_quat_wxyz)
    return quat_to_mat(rel)[:, :2].reshape(-1)