"""
参考动作数据加载与管理。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import G1_NUM_MOTOR, ISAACLAB_TO_MUJOCO, MUJOCO_TO_ISAACLAB


def read_csv_matrix(path):
    """读取 CSV 文件为 numpy 数组，支持带表头的 CSV"""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if data.dtype.names is not None:
        arr = np.column_stack([data[name] for name in data.dtype.names])
    else:
        arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def normalize_joint_order(joint_pos, joint_vel, joint_order):
    """
    将关节数据统一转换为 IsaacLab 顺序。
    如果输入已经是 isaaclab 顺序则直接返回；
    如果是 mujoco 顺序则通过映射表转换。
    """
    if joint_order == "isaaclab":
        return joint_pos, joint_vel
    if joint_order == "mujoco":
        return joint_pos[:, MUJOCO_TO_ISAACLAB], joint_vel[:, MUJOCO_TO_ISAACLAB]
    raise ValueError(f"Unsupported joint order: {joint_order}")


def load_soma_csv(csv_path, joint_order="isaaclab"):
    """
    加载单个 SOMA CSV 文件并转换为 ReferenceMotion 格式。
    
    SOMA CSV 格式:
    - Frame, root_translate{X,Y,Z}, root_rotate{X,Y,Z}, 29个关节自由度
    - 角度以度为单位，位置以厘米为单位
    - 关节顺序：MuJoCo/MJCF 执行器顺序
    
    Args:
        csv_path: CSV 文件路径
        joint_order: 期望的输出关节顺序 ("mujoco" 或 "isaaclab")，默认 "isaaclab"
    
    Returns:
        ReferenceMotion 对象
    """
    import pandas as pd
    from scipy.spatial import transform
    
    # 读取 CSV 数据
    data = pd.read_csv(csv_path)
    T = len(data)  # 时间步数
    
    # 根位置：厘米 → 米
    root_pos = (
        np.stack(
            [
                data["root_translateX"].values,
                data["root_translateY"].values,
                data["root_translateZ"].values,
            ],
            axis=1,
        ).astype(np.float64)
        / 100.0  # 厘米转米
    )
    
    # 根旋转：欧拉角xyz（内在）度 → 四元数（wxyz）
    euler_deg = np.stack(
        [
            data["root_rotateX"].values,
            data["root_rotateY"].values,
            data["root_rotateZ"].values,
        ],
        axis=1,
    ).astype(np.float64)
    root_quat_xyzw = (
        transform.Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat().astype(np.float64)
    )
    # 将xyzw → wxyz
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    
    # 关节自由度：度 → 弧度
    # 注意：CSV 中的关节顺序是 MuJoCo/MJCF 执行器顺序
    joint_cols = [c for c in data.columns if c.endswith("_dof")]
    if len(joint_cols) != G1_NUM_MOTOR:
        raise ValueError(
            f"Expected {G1_NUM_MOTOR} joint columns ending with '_dof', found {len(joint_cols)}: {joint_cols}"
        )
    
    joint_pos_mj = np.deg2rad(data[joint_cols].values).astype(np.float64)  # (T, 29) MuJoCo 顺序
    
    # 计算关节速度（通过差分）
    # 假设采样率为 50Hz (dt = 0.02s)
    joint_vel_mj = np.zeros_like(joint_pos_mj)
    joint_vel_mj[1:] = np.diff(joint_pos_mj, axis=0) / (1.0 / 50.0)
    joint_vel_mj[0] = joint_vel_mj[1]  # 复制第一帧
    
    # CSV 数据是 MuJoCo 顺序，需要根据 joint_order 参数转换
    # 如果 joint_order="isaaclab"（默认），需要 MuJoCo → IsaacLab
    # 如果 joint_order="mujoco"，保持不变
    if joint_order == "isaaclab":
        joint_pos_isaac = joint_pos_mj[:, MUJOCO_TO_ISAACLAB]
        joint_vel_isaac = joint_vel_mj[:, MUJOCO_TO_ISAACLAB]
    else:  # joint_order == "mujoco"
        joint_pos_isaac = joint_pos_mj
        joint_vel_isaac = joint_vel_mj
    
    # 创建 ReferenceMotion 对象
    return ReferenceMotion(
        name=Path(csv_path).stem,
        joint_pos_isaac=joint_pos_isaac,
        joint_vel_isaac=joint_vel_isaac,
        root_pos=root_pos,
        root_quat=root_quat_wxyz,
    )


@dataclass
class ReferenceMotion:
    """
    参考动作数据容器。
    所有内部数据均以 IsaacLab 关节顺序存储，
    提供 q_mujoco/dq_mujoco 方法按需转换为 MuJoCo 顺序。
    """
    name: str
    joint_pos_isaac: np.ndarray   # 关节位置 (N_frames x 29)
    joint_vel_isaac: np.ndarray   # 关节速度 (N_frames x 29)
    root_pos: np.ndarray          # 基座位置 (N_frames x 3)
    root_quat: np.ndarray         # 基座四元数 (N_frames x 4, wxyz)

    @property
    def frames(self):
        return self.joint_pos_isaac.shape[0]

    def frame_index(self, frame, loop):
        """获取帧索引，支持循环播放"""
        if loop:
            return frame % self.frames
        return min(frame, self.frames - 1)

    def q_mujoco(self, frame, loop):
        """获取指定帧的关节位置 (MuJoCo 顺序)"""
        return self.joint_pos_isaac[self.frame_index(frame, loop), ISAACLAB_TO_MUJOCO]

    def dq_mujoco(self, frame, loop):
        """获取指定帧的关节速度 (MuJoCo 顺序)"""
        return self.joint_vel_isaac[self.frame_index(frame, loop), ISAACLAB_TO_MUJOCO]


def resolve_motion_dir(path, motion_name=None):
    """
    解析参考动作目录路径。
    如果 path 本身包含 joint_pos.csv 则直接使用；
    否则在子目录中查找匹配的动作名称。
    """
    path = path.resolve()
    if (path / "joint_pos.csv").exists():
        return path
    candidates = [p for p in path.iterdir() if p.is_dir() and (p / "joint_pos.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"No motion folders with joint_pos.csv under {path}")
    if motion_name:
        for candidate in candidates:
            if candidate.name == motion_name:
                return candidate
        raise FileNotFoundError(f"Motion {motion_name!r} not found under {path}")
    return sorted(candidates)[0]


def load_reference_motion(path, motion_name=None, joint_order="isaaclab"):
    """
    从 CSV 文件或目录加载完整的参考动作数据。
    
    支持两种输入格式:
    1. SOMA CSV 单文件（包含 Frame, root_translate*, root_rotate*, *dof 列）
    2. 标准 reference 目录（包含 joint_pos.csv, joint_vel.csv 等）
    """
    path = path.resolve()
    
    # 检测是否为 CSV 文件
    if path.is_file() and path.suffix.lower() == ".csv":
        return load_soma_csv(path, joint_order)
    
    # 否则使用原有的目录加载逻辑
    motion_dir = resolve_motion_dir(path, motion_name)
    joint_pos = read_csv_matrix(motion_dir / "joint_pos.csv")
    joint_vel = read_csv_matrix(motion_dir / "joint_vel.csv")
    if joint_pos.shape[1] != G1_NUM_MOTOR or joint_vel.shape[1] != G1_NUM_MOTOR:
        raise ValueError(
            f"Expected 29 joint columns, got joint_pos={joint_pos.shape}, joint_vel={joint_vel.shape}"
        )
    # 统一转换为 IsaacLab 关节顺序
    joint_pos, joint_vel = normalize_joint_order(joint_pos, joint_vel, joint_order)

    # 加载基座位姿 (可选)
    body_pos_path = motion_dir / "body_pos.csv"
    body_quat_path = motion_dir / "body_quat.csv"
    if body_pos_path.exists():
        body_pos = read_csv_matrix(body_pos_path).reshape(joint_pos.shape[0], -1, 3)
        root_pos = body_pos[:, 0, :]  # 第一个 body 即为 root
    else:
        root_pos = np.zeros((joint_pos.shape[0], 3), dtype=np.float64)
        root_pos[:, 2] = 0.793  # 默认站立高度

    if body_quat_path.exists():
        body_quat = read_csv_matrix(body_quat_path).reshape(joint_pos.shape[0], -1, 4)
        root_quat = body_quat[:, 0, :]
    else:
        root_quat = np.zeros((joint_pos.shape[0], 4), dtype=np.float64)
        root_quat[:, 0] = 1.0  # 默认单位四元数

    return ReferenceMotion(
        name=motion_dir.name,
        joint_pos_isaac=joint_pos,
        joint_vel_isaac=joint_vel,
        root_pos=root_pos,
        root_quat=root_quat,
    )