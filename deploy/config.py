"""
SONIC 策略部署配置常量。

包含机器人参数、关节映射、PD 增益、动作缩放等配置。
"""

from pathlib import Path

import numpy as np


# ==================== 机器人常量定义 ====================
G1_NUM_MOTOR = 29       # Unitree G1 机器人电机总数
CONTROL_DT = 1.0 / 50.0 # 控制频率 50Hz (20ms)，与真机部署一致
SIM_DT = 1.0 / 500.0    # 物理仿真步长 500Hz (2ms)，保证数值稳定性

DEPLOY_ROOT = Path(__file__).resolve().parent

# ==================== 关节顺序映射表 ====================
# IsaacLab (训练环境) 与 MuJoCo (仿真/部署) 的关节索引映射
# 来源: policy_parameters.hpp
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
     11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)

# 默认站立姿态 (IsaacLab 顺序)
# 用于策略启动时的初始状态重置
DEFAULT_ANGLES = np.array(
    [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # 左腿
     -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # 右腿
     0.0, 0.0, 0.0,                           # 腰部
     0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # 左臂
     0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0],    # 右臂
    dtype=np.float64,
)

# ==================== PD 增益计算参数 ====================
# 基于电机转动惯量(armature)和期望的自然频率/阻尼比计算刚度(Kp)和阻尼(Kd)
# 公式: Kp = I * wn^2, Kd = 2 * zeta * I * wn
ARMATURE_5020 = 0.003609725      # 5020 电机转动惯量
ARMATURE_7520_14 = 0.010177520   # 7520-14 电机转动惯量
ARMATURE_7520_22 = 0.025101925   # 7520-22 电机转动惯量
ARMATURE_4010 = 0.00425          # 4010 电机转动惯量
NATURAL_FREQ = 10.0 * 2.0 * np.pi  # 自然频率 10Hz (rad/s)
DAMPING_RATIO = 2.0              # 阻尼比 (过阻尼，防止震荡)

# 各型号电机的基础刚度
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

# 各型号电机的基础阻尼
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# 各关节力矩上限 (Nm)，按 MuJoCo 关节顺序排列
TORQUE_LIMITS = np.array(
    [88, 88, 88, 139, 50, 50, 88, 88, 88, 139, 50, 50, 88, 50, 50,
     25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5],
    dtype=np.float64,
)

# 各关节 PD 刚度 (Kp)，按 MuJoCo 关节顺序
# 注意：部分关节使用了 2倍 基础刚度以增强跟踪性能
KPS = np.array(
    [STIFFNESS_7520_22, STIFFNESS_7520_22, STIFFNESS_7520_14,
     STIFFNESS_7520_22, 2 * STIFFNESS_5020, 2 * STIFFNESS_5020,
     STIFFNESS_7520_22, STIFFNESS_7520_22, STIFFNESS_7520_14,
     STIFFNESS_7520_22, 2 * STIFFNESS_5020, 2 * STIFFNESS_5020,
     STIFFNESS_7520_14, 2 * STIFFNESS_5020, 2 * STIFFNESS_5020,
     STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020,
     STIFFNESS_5020, STIFFNESS_4010, STIFFNESS_4010,
     STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020,
     STIFFNESS_5020, STIFFNESS_4010, STIFFNESS_4010],
    dtype=np.float64,
)

# 各关节 PD 阻尼 (Kd)，按 MuJoCo 关节顺序
KDS = np.array(
    [DAMPING_7520_22, DAMPING_7520_22, DAMPING_7520_14,
     DAMPING_7520_22, 2 * DAMPING_5020, 2 * DAMPING_5020,
     DAMPING_7520_22, DAMPING_7520_22, DAMPING_7520_14,
     DAMPING_7520_22, 2 * DAMPING_5020, 2 * DAMPING_5020,
     DAMPING_7520_14, 2 * DAMPING_5020, 2 * DAMPING_5020,
     DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020,
     DAMPING_5020, DAMPING_4010, DAMPING_4010,
     DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020,
     DAMPING_5020, DAMPING_4010, DAMPING_4010],
    dtype=np.float64,
)

# 策略动作缩放因子
# 将策略输出的归一化 action 转换为关节角度偏移量
# 公式: scale = 0.25 * torque_limit / stiffness
# 0.25 是经验系数，限制单步最大角度变化范围
ACTION_SCALE = np.array(
    [
        0.25 * 139.0 / STIFFNESS_7520_22,
        0.25 * 139.0 / STIFFNESS_7520_22,
        0.25 * 88.0 / STIFFNESS_7520_14,
        0.25 * 139.0 / STIFFNESS_7520_22,
        0.25 * 25.0 / STIFFNESS_5020,
        0.25 * 25.0 / STIFFNESS_5020,
        0.25 * 139.0 / STIFFNESS_7520_22,
        0.25 * 139.0 / STIFFNESS_7520_22,
        0.25 * 88.0 / STIFFNESS_7520_14,
        0.25 * 139.0 / STIFFNESS_7520_22,
        0.25 * 25.0 / STIFFNESS_5020,
        0.25 * 25.0 / STIFFNESS_5020,
        0.25 * 88.0 / STIFFNESS_7520_14,
        0.25 * 25.0 / STIFFNESS_5020,
        0.25 * 25.0 / STIFFNESS_5020,
        0.25 * 25.0 / STIFFNESS_5020,
        0.25 * 25.0 / STIFFNESS_5020,
        0.25 * 25.0 / STIFFNESS_5020,
        0.25 * 25.0 / STIFFNESS_5020,
        0.25 * 25.0 / STIFFNESS_5020,
        0.25 * 5.0 / STIFFNESS_4010,
        0.25 * 5.0 / STIFFNESS_4010,
        0.25 * 25.0 / STIFFNESS_5020,
        0.25 * 25.0 / STIFFNESS_5020,
        0.25 * 25.0 / STIFFNESS_5020,
        0.25 * 25.0 / STIFFNESS_5020,
        0.25 * 25.0 / STIFFNESS_5020,
        0.25 * 5.0 / STIFFNESS_4010,
        0.25 * 5.0 / STIFFNESS_4010,
    ],
    dtype=np.float64,
)