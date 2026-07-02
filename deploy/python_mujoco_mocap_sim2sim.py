"""
纯 Python 实现的 MuJoCo sim2sim 运行器，用于 SONIC 策略部署参考动作追踪。

本脚本刻意不依赖 unitree_mujoco、Unitree DDS 或 ROS，以保持轻量级和独立性。
它直接加载标准的 MuJoCo G1 MJCF 模型文件和部署侧的参考动作 CSV 文件夹。

运行模式 (Use modes):
  policy:    SONIC 编码器+解码器闭环控制，输出作为 MuJoCo PD 控制器的目标位置。
  kinematic: 仅用于诊断，直接播放 CSV 中的参考轨迹（无物理仿真）。
  pd:        仅用于诊断，使用参考关节角进行 PD 控制（无策略网络）。

高级功能:
  使用 --bake-playback 参数可以先在无头(headless)模式下运行策略生成轨迹缓存，
  然后在可视化器中直接回放缓存的 MuJoCo 轨迹，无需在渲染循环中实时运行 ONNX 推理。
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

import numpy as np

from config import (
    ACTION_SCALE,
    CONTROL_DT,
    DEFAULT_ANGLES,
    DEPLOY_ROOT,
    G1_NUM_MOTOR,
    ISAACLAB_TO_MUJOCO,
    KDS,
    KPS,
    MUJOCO_TO_ISAACLAB,
    SIM_DT,
    TORQUE_LIMITS,
)
from utils import (
    heading_quat,
    heading_quat_inv,
    quat_conj,
    quat_mul,
    quat_rotate_inv,
    quat_to_mat,
    rot6_from_quat_delta,
)


# ==================== 数据加载工具函数 ====================

def read_csv_matrix(path: Path) -> np.ndarray:
    """读取 CSV 文件为 numpy 数组，支持带表头的 CSV"""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if data.dtype.names is not None:
        arr = np.column_stack([data[name] for name in data.dtype.names])
    else:
        arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def normalize_joint_order(
    joint_pos: np.ndarray, joint_vel: np.ndarray, joint_order: str
) -> tuple[np.ndarray, np.ndarray]:
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
    def frames(self) -> int:
        return self.joint_pos_isaac.shape[0]

    def frame_index(self, frame: int, loop: bool) -> int:
        """获取帧索引，支持循环播放"""
        if loop:
            return frame % self.frames
        return min(frame, self.frames - 1)

    def q_mujoco(self, frame: int, loop: bool) -> np.ndarray:
        """获取指定帧的关节位置 (MuJoCo 顺序)"""
        return self.joint_pos_isaac[self.frame_index(frame, loop), ISAACLAB_TO_MUJOCO]

    def dq_mujoco(self, frame: int, loop: bool) -> np.ndarray:
        """获取指定帧的关节速度 (MuJoCo 顺序)"""
        return self.joint_vel_isaac[self.frame_index(frame, loop), ISAACLAB_TO_MUJOCO]


def resolve_motion_dir(path: Path, motion_name: str | None) -> Path:
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


def load_reference_motion(
    path: Path,
    motion_name: str | None = None,
    joint_order: str = "isaaclab",
) -> ReferenceMotion:
    """
    从 CSV 文件加载完整的参考动作数据。
    必需文件: joint_pos.csv, joint_vel.csv
    可选文件: body_pos.csv, body_quat.csv (若缺失则使用默认值)
    """
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


class OrtRunner:
    """
    ONNX Runtime 推理封装器。
    自动处理输入维度不匹配的情况（填充或截断），并打印警告。
    """
    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        self.expected_dim = self._read_expected_dim(self.input_shape)
        self._shape_warning_printed = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        # 处理输入维度不匹配
        if self.expected_dim is not None and x.size != self.expected_dim:
            if x.size < self.expected_dim:
                x = np.pad(x, (0, self.expected_dim - x.size))
                action = "padded"
            else:
                x = x[: self.expected_dim]
                action = "truncated"
            if not self._shape_warning_printed:
                print(f"Warning: {self.input_name} was {action} to ONNX input dim {self.expected_dim}.")
                self._shape_warning_printed = True
        y = self.session.run([self.output_name], {self.input_name: x.reshape(1, -1)})[0]
        return np.asarray(y, dtype=np.float32).reshape(-1)

    @staticmethod
    def _read_expected_dim(shape: list[object]) -> int | None:
        if not shape:
            return None
        dim = shape[-1]
        return dim if isinstance(dim, int) and dim > 0 else None


@dataclass
class RobotSnapshot:
    """
    机器人状态快照，用于构建观测历史。
    存储 MuJoCo 原始数据和上一次策略动作。
    """
    base_quat: np.ndarray       # 基座四元数 (wxyz)
    base_ang_vel: np.ndarray    # 基座角速度 (陀螺仪)
    q_mujoco: np.ndarray        # 关节位置 (MuJoCo 顺序)
    dq_mujoco: np.ndarray       # 关节速度 (MuJoCo 顺序)
    last_action_isaac: np.ndarray  # 上次策略输出 (IsaacLab 顺序)

    @property
    def body_q_isaac(self) -> np.ndarray:
        """转换为 IsaacLab 顺序并减去默认姿态 (得到相对偏移量)"""
        return self.q_mujoco[MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB]

    @property
    def body_dq_isaac(self) -> np.ndarray:
        """关节速度转换为 IsaacLab 顺序"""
        return self.dq_mujoco[MUJOCO_TO_ISAACLAB]

    @property
    def gravity_dir(self) -> np.ndarray:
        """重力方向在基座局部坐标系中的表示"""
        return quat_rotate_inv(self.base_quat, np.array([0.0, 0.0, -1.0]))


class ObservationBuilder:
    """
    SONIC 策略观测构建器。
    负责组装编码器输入（参考动作片段）和解码器输入（历史状态+token）。
    """
    def __init__(
        self,
        reference: ReferenceMotion,
        encoder: Callable[[np.ndarray], np.ndarray] | None,
        encoder_input_dim: int | None = None,
        future_step: int = 5,      # 参考动作采样间隔 (帧)
        history_frames: int = 10,  # 历史状态窗口长度
    ) -> None:
        self.reference = reference
        self.encoder = encoder
        self.encoder_input_dim = encoder_input_dim
        self.future_step = future_step
        self.history_frames = history_frames
        self.history: deque[RobotSnapshot] = deque(maxlen=256)
        self.token_state = np.zeros(64, dtype=np.float64)  # 编码器输出的 latent token
        self.init_base_quat: np.ndarray | None = None      # 初始基座朝向 (用于对齐参考动作)
        self.init_ref_quat = reference.root_quat[0].copy()

    def push_snapshot(self, snapshot: RobotSnapshot) -> None:
        """记录新的状态快照"""
        if self.init_base_quat is None:
            self.init_base_quat = snapshot.base_quat.copy()
        self.history.append(snapshot)

    def reset_episode(self) -> None:
        """重置 episode 状态"""
        self.history.clear()
        self.token_state.fill(0.0)
        self.init_base_quat = None

    def policy_observation(self, frame: int, loop: bool) -> np.ndarray:
        """
        构建完整的策略(解码器)观测向量。
        包含: token状态 + 历史角速度 + 历史关节位置 + 历史关节速度 + 历史动作 + 历史重力方向
        """
        parts = [
            self.token_state_observation(frame, loop),
            self.history_values("base_ang_vel", self.history_frames),
            self.history_values("body_q_isaac", self.history_frames),
            self.history_values("body_dq_isaac", self.history_frames),
            self.history_values("last_action_isaac", self.history_frames),
            self.history_values("gravity_dir", self.history_frames),
        ]
        return np.concatenate(parts).astype(np.float32)

    def token_state_observation(self, frame: int, loop: bool) -> np.ndarray:
        """
        获取/更新编码器输出的 token 状态。
        每次调用都会用当前参考动作片段重新编码。
        """
        if self.encoder is None:
            return self.token_state.copy()
        self.token_state = self.encoder(self.encoder_observation(frame, loop)).astype(np.float64)
        return self.token_state.copy()

    def encoder_observation(self, frame: int, loop: bool) -> np.ndarray:
        """构建编码器输入观测"""
        if self.encoder_input_dim == 1762:
            return self.encoder_observation_legacy1762(frame, loop)

        # 标准格式: [encoder_index(1) + command(580) + anchor(60)]
        encoder_index = np.array([0.0], dtype=np.float64)
        command = self.motion_joint_pos_vel(frame, loop, self.history_frames, self.future_step)
        anchor = self.motion_anchor_orientation(frame, loop, self.history_frames, self.future_step)
        return np.concatenate([encoder_index, command, anchor]).astype(np.float32)

    def encoder_observation_legacy1762(self, frame: int, loop: bool) -> np.ndarray:
        """
        兼容旧版 1762 维编码器输入格式。
        对应 C++ GatherEncoderObservations() 的内存布局:
          offset 0:   encoder_mode_4 (4D, 全零)
          offset 4:   motion_joint_positions_10frame_step5 (290D)
          offset 294: motion_joint_velocities_10frame_step5 (290D)
          offset 601: motion_anchor_orientation_10frame_step5 (60D)
        其余位置补零。
        """
        obs = np.zeros(1762, dtype=np.float32)
        obs[0:4] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs[4:294] = self.motion_joint_positions(frame, loop, 10, 5).astype(np.float32)
        obs[294:584] = self.motion_joint_velocities(frame, loop, 10, 5).astype(np.float32)
        obs[601:661] = self.motion_anchor_orientation(frame, loop, 10, 5).astype(np.float32)
        return obs

    def history_values(self, attr: str, frames: int) -> np.ndarray:
        """
        从历史缓冲区中提取指定属性的最近 N 帧数据。
        不足的部分用零填充。
        """
        if not self.history:
            raise RuntimeError("No robot snapshot in history")
        values: list[np.ndarray] = []
        hist = list(self.history)
        missing = max(0, frames - len(hist))
        if missing:
            values.extend(self.zero_history_value(attr) for _ in range(missing))
        for src in hist[-frames:]:
            values.append(np.asarray(getattr(src, attr), dtype=np.float64).reshape(-1))
        return np.concatenate(values)

    @staticmethod
    def zero_history_value(attr: str) -> np.ndarray:
        """生成指定属性的零值占位符"""
        if attr == "base_ang_vel":
            return np.zeros(3, dtype=np.float64)
        if attr in ("body_q_isaac", "body_dq_isaac", "last_action_isaac"):
            return np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        if attr == "gravity_dir":
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        raise ValueError(f"Unsupported history attr: {attr}")

    def motion_joint_pos_vel(self, frame: int, loop: bool, frames: int, step: int) -> np.ndarray:
        """获取未来多帧的关节位置和速度 (交错排列)"""
        rows = []
        for i in range(frames):
            idx = self.reference.frame_index(frame + i * step, loop)
            rows.append(self.reference.joint_pos_isaac[idx])
            rows.append(self.reference.joint_vel_isaac[idx])
        return np.concatenate(rows)

    def motion_joint_positions(
        self, frame: int, loop: bool, frames: int, step: int,
        joint_indexes: np.ndarray | None = None,
    ) -> np.ndarray:
        """获取未来多帧的关节位置"""
        rows = []
        for i in range(frames):
            idx = self.reference.frame_index(frame + i * step, loop)
            value = self.reference.joint_pos_isaac[idx]
            rows.append(value if joint_indexes is None else value[joint_indexes])
        return np.concatenate(rows)

    def motion_joint_velocities(
        self, frame: int, loop: bool, frames: int, step: int,
        joint_indexes: np.ndarray | None = None,
    ) -> np.ndarray:
        """获取未来多帧的关节速度"""
        rows = []
        for i in range(frames):
            idx = self.reference.frame_index(frame + i * step, loop)
            value = self.reference.joint_vel_isaac[idx]
            rows.append(value if joint_indexes is None else value[joint_indexes])
        return np.concatenate(rows)

    def motion_anchor_orientation(self, frame: int, loop: bool, frames: int, step: int) -> np.ndarray:
        """
        获取未来多帧的锚点朝向 (6D 旋转表示)。
        关键逻辑: 将参考动作的朝向对齐到机器人初始朝向，
        消除全局偏航差异，使策略只关注相对运动。
        """
        base = self.history[-1].base_quat if self.history else np.array([1.0, 0.0, 0.0, 0.0])
        init_base = self.init_base_quat if self.init_base_quat is not None else base
        # 计算初始时刻机器人与参考动作之间的偏航差
        apply_delta_heading = quat_mul(heading_quat(init_base), heading_quat_inv(self.init_ref_quat))
        rows = []
        for i in range(frames):
            idx = self.reference.frame_index(frame + i * step, loop)
            # 将参考朝向应用偏航对齐后，计算相对于当前基座的 6D 旋转
            ref_quat = quat_mul(apply_delta_heading, self.reference.root_quat[idx])
            rows.append(rot6_from_quat_delta(base, ref_quat))
        return np.concatenate(rows)


class MujocoG1Sim:
    """
    MuJoCo G1 仿真封装。
    管理模型加载、状态读写、PD 控制和物理步进。
    """
    def __init__(self, xml_path: Path, sim_dt: float = SIM_DT) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = sim_dt

        # 预计算每个电机对应的 qpos/qvel 地址
        # actuator_trnid[i,0] 是第 i 个执行器关联的关节 ID
        self.qpos_addrs = np.zeros(G1_NUM_MOTOR, dtype=np.int64)
        self.qvel_addrs = np.zeros(G1_NUM_MOTOR, dtype=np.int64)
        for i in range(G1_NUM_MOTOR):
            joint_id = int(self.model.actuator_trnid[i, 0])
            self.qpos_addrs[i] = self.model.jnt_qposadr[joint_id]
            self.qvel_addrs[i] = self.model.jnt_dofadr[joint_id]

        # 查找 IMU 陀螺仪传感器地址
        self.gyro_sensor_adr: int | None = None
        self.gyro_sensor_dim = 0
        sensor_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_SENSOR, "imu-angular-velocity"
        )
        if sensor_id >= 0:
            self.gyro_sensor_adr = int(self.model.sensor_adr[sensor_id])
            self.gyro_sensor_dim = int(self.model.sensor_dim[sensor_id])

    def reset_to_reference(self, reference: ReferenceMotion) -> None:
        """重置仿真到参考动作的第一帧"""
        self.mujoco.mj_resetData(self.model, self.data)
        self.set_kinematic_reference(reference, 0, loop=False)

    def reset_to_default_pose(self, root_height: float) -> None:
        """重置到默认站立姿态 (策略启动常用)"""
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = np.array([0.0, 0.0, root_height], dtype=np.float64)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.set_joint_qpos(DEFAULT_ANGLES)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:G1_NUM_MOTOR] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def set_joint_qpos(self, q: np.ndarray) -> None:
        """设置所有电机关节位置"""
        self.data.qpos[self.qpos_addrs] = q

    def set_joint_qvel(self, dq: np.ndarray) -> None:
        """设置所有电机关节速度"""
        self.data.qvel[self.qvel_addrs] = dq

    def q_mujoco(self) -> np.ndarray:
        """读取当前关节位置"""
        return self.data.qpos[self.qpos_addrs].copy()

    def dq_mujoco(self) -> np.ndarray:
        """读取当前关节速度"""
        return self.data.qvel[self.qvel_addrs].copy()

    def base_quat(self) -> np.ndarray:
        """读取基座四元数 (free joint 的前 4 个 qpos 分量)"""
        return self.data.qpos[3:7].copy()

    def base_ang_vel(self) -> np.ndarray:
        """
        读取基座角速度。
        优先使用 IMU 陀螺仪传感器数据（更接近真实部署），
        回退到 free joint 的 qvel。
        """
        if self.gyro_sensor_adr is not None and self.gyro_sensor_dim == 3:
            adr = self.gyro_sensor_adr
            return self.data.sensordata[adr : adr + 3].copy()
        return self.data.qvel[3:6].copy()

    def set_root_pose(self, reference: ReferenceMotion, frame: int, loop: bool) -> None:
        """设置基座位姿到参考动作指定帧"""
        idx = reference.frame_index(frame, loop)
        self.data.qpos[0:3] = reference.root_pos[idx]
        self.data.qpos[3:7] = reference.root_quat[idx]

    def pin_root_to_reference(self, reference: ReferenceMotion, frame: int, loop: bool) -> None:
        """
        将基座强制固定到参考动作位姿 (诊断用)。
        同时清零基座速度，实现完全运动学约束。
        """
        self.set_root_pose(reference, frame, loop)
        self.data.qvel[0:6] = 0.0

    def set_kinematic_reference(self, reference: ReferenceMotion, frame: int, loop: bool) -> None:
        """完全运动学设置 (位置+速度+基座)，用于 kinematic 模式"""
        self.set_root_pose(reference, frame, loop)
        self.set_joint_qpos(reference.q_mujoco(frame, loop))
        self.set_joint_qvel(reference.dq_mujoco(frame, loop))
        self.data.ctrl[:G1_NUM_MOTOR] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def apply_pd(self, q_target: np.ndarray, dq_target: np.ndarray, kp_scale: float, kd_scale: float) -> None:
        """
        应用 PD 控制器计算力矩并写入 ctrl。
        tau = Kp * scale * (q_target - q) + Kd * scale * (dq_target - dq)
        结果裁剪到力矩限制范围内。
        """
        q = self.q_mujoco()
        dq = self.dq_mujoco()
        tau = (KPS * kp_scale) * (q_target - q) + (KDS * kd_scale) * (dq_target - dq)
        self.data.ctrl[:G1_NUM_MOTOR] = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)

    def step(self, n: int = 1) -> None:
        """执行 n 步物理仿真"""
        for _ in range(n):
            self.mujoco.mj_step(self.model, self.data)


def policy_target_from_action(action_isaac: np.ndarray) -> np.ndarray:
    """
    将策略输出的 action (IsaacLab 顺序) 转换为 MuJoCo 关节目标位置。
    target = default_pose + action * scale
    """
    if action_isaac.size != G1_NUM_MOTOR:
        raise RuntimeError(f"Policy action must be 29D, got {action_isaac.size}")
    return DEFAULT_ANGLES + action_isaac[ISAACLAB_TO_MUJOCO] * ACTION_SCALE


def compute_target(
    args: argparse.Namespace,
    reference: ReferenceMotion,
    obs_builder: ObservationBuilder,
    policy: Callable[[np.ndarray], np.ndarray] | None,
    frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据控制器模式计算目标关节位置/速度和策略动作。
    返回: (q_target, dq_target, action_isaac)
    """
    if args.controller in ("kinematic", "pd"):
        return (
            reference.q_mujoco(frame, args.reference_loop),
            reference.dq_mujoco(frame, args.reference_loop),
            np.zeros(G1_NUM_MOTOR, dtype=np.float64),
        )

    if policy is None:
        raise RuntimeError("Policy controller requested but no policy runner was created")
    
    # 构建观测 → 策略推理 → 可选裁剪
    obs = obs_builder.policy_observation(frame, args.reference_loop)
    action_isaac = policy(obs).astype(np.float64)
    if args.action_clip > 0.0:
        action_isaac = np.clip(action_isaac, -args.action_clip, args.action_clip)
    return policy_target_from_action(action_isaac), np.zeros(G1_NUM_MOTOR), action_isaac


def maybe_print_metrics(args: argparse.Namespace, reference: ReferenceMotion, sim: MujocoG1Sim, frame: int) -> None:
    """定期打印关节跟踪误差指标"""
    if args.print_every <= 0 or frame % args.print_every != 0:
        return
    q_ref = reference.q_mujoco(frame, args.reference_loop)
    err = sim.q_mujoco() - q_ref
    prefix = f"episode={args.episode:03d} " if args.episode_loop else ""
    print(
        f"{prefix}frame={frame:05d} "
        f"joint_rmse={np.sqrt(np.mean(err**2)):.4f} "
        f"joint_max={np.max(np.abs(err)):.4f}"
    )


def run(args: argparse.Namespace) -> None:
    """主运行函数，处理各种运行模式的调度"""
    try:
        import mujoco.viewer
    except Exception as exc:
        raise RuntimeError("Install mujoco first: pip install mujoco") from exc

    # === 模式1: 加载缓存轨迹并回放 ===
    if args.load_rollout is not None:
        rollout = np.load(args.load_rollout)
        qpos_traj = np.asarray(rollout["qpos"], dtype=np.float64)
        qvel_traj = np.asarray(rollout["qvel"], dtype=np.float64)
        motion_name = (
            str(np.asarray(rollout["motion_name"]).item()) if "motion_name" in rollout else args.load_rollout.name
        )
        if args.headless:
            print(f"Loaded cached rollout {args.load_rollout}: qpos={qpos_traj.shape}, qvel={qvel_traj.shape}")
            return
        sim = MujocoG1Sim(args.xml, args.sim_dt)
        play_cached_rollout(args, sim, qpos_traj, qvel_traj, motion_name)
        return

    # 加载参考动作
    reference = load_reference_motion(args.motion_data, args.motion_name, args.joint_order)

    # === 模式2: 烘焙(bake)轨迹后回放 ===
    if args.bake_playback:
        qpos_traj, qvel_traj = bake_rollout(args, reference)
        if args.save_rollout is not None:
            save_rollout(args.save_rollout, args, reference.name, qpos_traj, qvel_traj)
        if args.headless:
            print(f"Headless bake complete: qpos={qpos_traj.shape}, qvel={qvel_traj.shape}")
            return
        sim = MujocoG1Sim(args.xml, args.sim_dt)
        play_cached_rollout(args, sim, qpos_traj, qvel_traj, reference.name)
        return

    # === 模式3: 实时仿真 (policy/kinematic/pd) ===
    sim = MujocoG1Sim(args.xml, args.sim_dt)
    reset_sim_for_controller(args, reference, sim)

    encoder = None
    policy = None
    encoder_input_dim = None
    if args.controller == "policy":
        encoder = OrtRunner(args.encoder)
        policy = OrtRunner(args.policy)
        encoder_input_dim = encoder.expected_dim
        print(f"Loaded encoder {args.encoder} input_shape={encoder.input_shape}")
        print(f"Loaded policy  {args.policy} input_shape={policy.input_shape}")

    obs_builder = ObservationBuilder(
        reference=reference,
        encoder=encoder,
        encoder_input_dim=encoder_input_dim,
        future_step=args.future_step,
        history_frames=args.history_frames,
    )

    last_action = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
    frame = 0
    sim_steps_per_control = max(1, round(CONTROL_DT / args.sim_dt))
    
    # 判断是否无限循环运行
    viewer_runs_forever = (not args.headless) and args.max_steps == 0
    args.reference_loop = args.loop and args.controller == "kinematic"
    args.episode_loop = args.loop and args.controller != "kinematic"
    if viewer_runs_forever:
        if args.controller == "kinematic":
            args.reference_loop = True
        else:
            args.episode_loop = True
        max_steps: int | None = None
    else:
        max_steps = args.max_steps if args.max_steps > 0 else reference.frames
    
    # 启动可视化器 (非 headless 模式)
    viewer_cm = None if args.headless else mujoco.viewer.launch_passive(sim.model, sim.data)
    args.episode = 0

    print(
        f"Tracking {reference.name}: frames={reference.frames}, controller={args.controller}, "
        f"joint_order={args.joint_order}, reference_loop={args.reference_loop}, "
        f"episode_loop={args.episode_loop}, xml={args.xml}"
    )
    if viewer_runs_forever:
        print("Viewer runs until Ctrl+C.")
    else:
        print("Press Ctrl+C to stop.")

    try:
        if viewer_cm is None:
            # Headless 模式
            if max_steps is None:
                raise RuntimeError("Internal error: headless runs must have finite max_steps")
            for _ in range(max_steps):
                last_action = run_control_tick(
                    args, reference, sim, obs_builder, policy, last_action, frame, sim_steps_per_control
                )
                maybe_print_metrics(args, reference, sim, frame)
                frame += 1
                if args.episode_loop and frame >= reference.frames:
                    frame = 0
                    args.episode += 1
                    last_action = reset_episode_state(args, reference, sim, obs_builder)
        else:
            # 带可视化模式
            with viewer_cm as viewer:
                while viewer.is_running() and (max_steps is None or frame < max_steps):
                    tick = time.perf_counter()
                    last_action = run_control_tick(
                        args, reference, sim, obs_builder, policy, last_action, frame, sim_steps_per_control
                    )
                    viewer.sync()
                    maybe_print_metrics(args, reference, sim, frame)
                    frame += 1
                    if args.episode_loop and frame >= reference.frames:
                        frame = 0
                        args.episode += 1
                        last_action = reset_episode_state(args, reference, sim, obs_builder)
                    # 实时限速
                    if args.realtime:
                        sleep_s = CONTROL_DT - (time.perf_counter() - tick)
                        if sleep_s > 0:
                            time.sleep(sleep_s)
    except KeyboardInterrupt:
        pass


def bake_rollout(
    args: argparse.Namespace,
    reference: ReferenceMotion,
) -> tuple[np.ndarray, np.ndarray]:
    """
    无头模式下运行完整策略推理，缓存所有帧的 qpos/qvel。
    避免在可视化循环中重复运行 ONNX，提高回放流畅度。
    """
    sim = MujocoG1Sim(args.xml, args.sim_dt)
    reset_sim_for_controller(args, reference, sim)

    encoder = None
    policy = None
    encoder_input_dim = None
    if args.controller == "policy":
        encoder = OrtRunner(args.encoder)
        policy = OrtRunner(args.policy)
        encoder_input_dim = encoder.expected_dim
        print(f"Loaded encoder {args.encoder} input_shape={encoder.input_shape}")
        print(f"Loaded policy  {args.policy} input_shape={policy.input_shape}")

    obs_builder = ObservationBuilder(
        reference=reference,
        encoder=encoder,
        encoder_input_dim=encoder_input_dim,
        future_step=args.future_step,
        history_frames=args.history_frames,
    )

    max_steps = args.max_steps if args.max_steps > 0 else reference.frames
    sim_steps_per_control = max(1, round(CONTROL_DT / args.sim_dt))
    args.reference_loop = args.loop and args.controller == "kinematic"
    args.episode_loop = args.loop and args.controller != "kinematic"
    args.episode = 0

    print(
        f"Baking {reference.name}: frames={reference.frames}, steps={max_steps}, "
        f"controller={args.controller}, xml={args.xml}"
    )
    qpos_traj: list[np.ndarray] = []
    qvel_traj: list[np.ndarray] = []
    last_action = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
    frame = 0
    for _ in range(max_steps):
        last_action = run_control_tick(
            args, reference, sim, obs_builder, policy, last_action, frame, sim_steps_per_control
        )
        qpos_traj.append(sim.data.qpos.copy())
        qvel_traj.append(sim.data.qvel.copy())
        maybe_print_metrics(args, reference, sim, frame)
        frame += 1
        if args.episode_loop and frame >= reference.frames:
            frame = 0
            args.episode += 1
            last_action = reset_episode_state(args, reference, sim, obs_builder)

    print(f"Baked {len(qpos_traj)} control frames.")
    return np.stack(qpos_traj), np.stack(qvel_traj)


def save_rollout(
    path: Path,
    args: argparse.Namespace,
    motion_name: str,
    qpos_traj: np.ndarray,
    qvel_traj: np.ndarray,
) -> None:
    """保存烘焙的轨迹到 .npz 文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        qpos=qpos_traj,
        qvel=qvel_traj,
        motion_name=np.array(motion_name),
        controller=np.array(args.controller),
        control_dt=np.array(CONTROL_DT),
        sim_dt=np.array(args.sim_dt),
        xml=np.array(str(args.xml)),
    )
    print(f"Saved baked rollout to {path}")


def play_cached_rollout(
    args: argparse.Namespace,
    sim: MujocoG1Sim,
    qpos_traj: np.ndarray,
    qvel_traj: np.ndarray,
    motion_name: str,
) -> None:
    """在可视化器中回放缓存的轨迹 (不运行 ONNX)"""
    import mujoco.viewer

    if qpos_traj.ndim != 2 or qpos_traj.shape[1] != sim.model.nq:
        raise ValueError(f"qpos rollout shape {qpos_traj.shape} does not match model.nq={sim.model.nq}")
    if qvel_traj.ndim != 2 or qvel_traj.shape[1] != sim.model.nv:
        raise ValueError(f"qvel rollout shape {qvel_traj.shape} does not match model.nv={sim.model.nv}")
    if qpos_traj.shape[0] != qvel_traj.shape[0]:
        raise ValueError("qpos and qvel rollout lengths differ")

    print(
        f"Playing cached rollout {motion_name}: frames={qpos_traj.shape[0]}, "
        f"loop={args.playback_loop}, xml={args.xml}"
    )
    print("Cached viewer does not run ONNX. Ctrl+C or close the viewer to stop.")

    frame = 0
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            tick = time.perf_counter()
            if frame >= qpos_traj.shape[0]:
                if not args.playback_loop:
                    break
                frame = 0
            # 直接设置状态而非施加力矩
            sim.data.qpos[:] = qpos_traj[frame]
            sim.data.qvel[:] = qvel_traj[frame]
            sim.data.ctrl[:G1_NUM_MOTOR] = 0.0
            sim.mujoco.mj_forward(sim.model, sim.data)
            viewer.sync()
            frame += 1
            if args.realtime:
                sleep_s = CONTROL_DT - (time.perf_counter() - tick)
                if sleep_s > 0:
                    time.sleep(sleep_s)


def run_control_tick(
    args: argparse.Namespace,
    reference: ReferenceMotion,
    sim: MujocoG1Sim,
    obs_builder: ObservationBuilder,
    policy: Callable[[np.ndarray], np.ndarray] | None,
    last_action: np.ndarray,
    frame: int,
    sim_steps_per_control: int,
) -> np.ndarray:
    """
    执行一个控制周期的完整流程:
    1. 记录状态快照
    2. 计算目标 (策略推理或参考轨迹)
    3. 执行多个物理仿真子步 (PD 控制)
    返回本次的策略动作 (用于下次观测的历史输入)
    """
    if args.controller == "kinematic":
        sim.set_kinematic_reference(reference, frame, args.reference_loop)
        return np.zeros(G1_NUM_MOTOR, dtype=np.float64)

    # 记录当前状态到历史缓冲区
    snapshot = RobotSnapshot(
        base_quat=sim.base_quat(),
        base_ang_vel=sim.base_ang_vel(),
        q_mujoco=sim.q_mujoco(),
        dq_mujoco=sim.dq_mujoco(),
        last_action_isaac=last_action.copy(),
    )
    obs_builder.push_snapshot(snapshot)
    
    # 计算目标关节位置
    q_target, dq_target, next_action = compute_target(args, reference, obs_builder, policy, frame)
    
    # 在每个控制周期内执行多个物理子步
    pin_root = args.pin_root
    if pin_root:
        for _ in range(sim_steps_per_control):
            sim.pin_root_to_reference(reference, frame, args.reference_loop)
            sim.apply_pd(q_target, dq_target, args.kp_scale, args.kd_scale)
            sim.step(1)
    else:
        for _ in range(sim_steps_per_control):
            sim.apply_pd(q_target, dq_target, args.kp_scale, args.kd_scale)
            sim.step(1)
    return next_action


def reset_sim_for_controller(
    args: argparse.Namespace,
    reference: ReferenceMotion,
    sim: MujocoG1Sim,
) -> None:
    """根据控制器类型选择重置方式"""
    if args.controller == "policy" and args.policy_start_pose == "default":
        sim.reset_to_default_pose(args.root_height)
    else:
        sim.reset_to_reference(reference)


def reset_episode_state(
    args: argparse.Namespace,
    reference: ReferenceMotion,
    sim: MujocoG1Sim,
    obs_builder: ObservationBuilder,
) -> np.ndarray:
    """重置 episode 状态 (仿真+观测构建器)"""
    reset_sim_for_controller(args, reference, sim)
    obs_builder.reset_episode()
    return np.zeros(G1_NUM_MOTOR, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEPLOY_ROOT / "g1" / "scene_29dof.xml")
    parser.add_argument("--motion-data", type=Path, default=DEPLOY_ROOT / "reference" / "example")
    parser.add_argument("--motion-name", default="")
    parser.add_argument(
        "--controller",
        choices=["policy", "kinematic", "pd"],
        default="policy",
        help="policy runs SONIC encoder/decoder closed loop; kinematic/pd are diagnostics.",
    )
    parser.add_argument(
        "--joint-order",
        choices=["isaaclab", "mujoco"],
        default="isaaclab",
        help="Deploy reference folders use isaaclab order. Target-motion logs are usually mujoco order.",
    )
    parser.add_argument("--encoder", type=Path, default=DEPLOY_ROOT / "policy" / "release" / "model_encoder.onnx")
    parser.add_argument("--policy", type=Path, default=DEPLOY_ROOT / "policy" / "release" / "model_decoder.onnx")
    parser.add_argument("--history-frames", type=int, default=10, help="历史状态窗口帧数")
    parser.add_argument("--future-step", type=int, default=5, help="参考动作采样步长")
    parser.add_argument("--sim-dt", type=float, default=SIM_DT, help="MuJoCo 物理仿真步长")
    parser.add_argument("--kp-scale", type=float, default=1.0, help="PD 刚度缩放因子")
    parser.add_argument("--kd-scale", type=float, default=1.0, help="PD 阻尼缩放因子")
    parser.add_argument(
        "--action-clip",
        type=float,
        default=0.0,
        help="策略动作裁剪阈值，0 表示不裁剪",
    )
    parser.add_argument(
        "--policy-start-pose",
        choices=["default", "reference"],
        default="default",
        help="策略启动时的初始姿态",
    )
    parser.add_argument("--root-height", type=float, default=0.793, help="默认站立高度")
    parser.add_argument("--pin-root", action="store_true", help="诊断用: 将基座固定到参考轨迹")
    parser.add_argument(
        "--bake-playback",
        action="store_true",
        help="先无头烘焙轨迹再可视化回放",
    )
    parser.add_argument("--save-rollout", type=Path, default=None, help="保存烘焙轨迹的路径 (.npz)")
    parser.add_argument("--load-rollout", type=Path, default=None, help="加载已烘焙轨迹并回放")
    parser.add_argument("--no-playback-loop", dest="playback_loop", action="store_false")
    parser.set_defaults(playback_loop=True)
    parser.add_argument("--loop", action="store_true", help="循环播放参考动作")
    parser.add_argument("--headless", action="store_true", help="无头模式 (不显示可视化器)")
    parser.add_argument("--no-realtime", dest="realtime", action="store_false")
    parser.set_defaults(realtime=True)
    parser.add_argument("--max-steps", type=int, default=0, help="最大步数，0 表示运行完整动作")
    parser.add_argument("--print-every", type=int, default=50, help="每隔多少帧打印一次指标")
    args = parser.parse_args()
    args.motion_name = args.motion_name or None
    args.xml = args.xml.resolve()
    args.motion_data = args.motion_data.resolve()
    args.encoder = args.encoder.resolve()
    args.policy = args.policy.resolve()
    if args.save_rollout is not None:
        args.save_rollout = args.save_rollout.resolve()
    if args.load_rollout is not None:
        args.load_rollout = args.load_rollout.resolve()
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()