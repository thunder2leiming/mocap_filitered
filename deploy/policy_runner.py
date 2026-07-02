"""
策略推理与闭环控制。
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from config import (
    ACTION_SCALE,
    CONTROL_DT,
    DEFAULT_ANGLES,
    G1_NUM_MOTOR,
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
)
from motion_data import load_reference_motion, ReferenceMotion
from onnx_runner import OrtRunner
from simulator import MujocoG1Sim
from utils import (
    heading_quat,
    heading_quat_inv,
    quat_mul,
    quat_rotate_inv,
    rot6_from_quat_delta,
)


@dataclass
class RobotSnapshot:
    base_quat: np.ndarray
    base_ang_vel: np.ndarray
    q_mujoco: np.ndarray
    dq_mujoco: np.ndarray
    last_action_isaac: np.ndarray

    @property
    def body_q_isaac(self):
        return self.q_mujoco[MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB]

    @property
    def body_dq_isaac(self):
        return self.dq_mujoco[MUJOCO_TO_ISAACLAB]

    @property
    def gravity_dir(self):
        return quat_rotate_inv(self.base_quat, np.array([0.0, 0.0, -1.0]))


class ObservationBuilder:
    def __init__(self, reference, encoder, encoder_input_dim, future_step=5, history_frames=10):
        self.reference = reference
        self.encoder = encoder
        self.encoder_input_dim = encoder_input_dim
        self.future_step = future_step
        self.history_frames = history_frames
        self.history = deque(maxlen=256)
        self.token_state = np.zeros(64, dtype=np.float64)
        self.init_base_quat = None
        self.init_ref_quat = reference.root_quat[0].copy()

    def push_snapshot(self, snapshot):
        if self.init_base_quat is None:
            self.init_base_quat = snapshot.base_quat.copy()
        self.history.append(snapshot)

    def reset_episode(self):
        self.history.clear()
        self.token_state.fill(0.0)
        self.init_base_quat = None

    def policy_observation(self, frame, loop):
        parts = [
            self.token_state_observation(frame, loop),
            self.history_values("base_ang_vel", self.history_frames),
            self.history_values("body_q_isaac", self.history_frames),
            self.history_values("body_dq_isaac", self.history_frames),
            self.history_values("last_action_isaac", self.history_frames),
            self.history_values("gravity_dir", self.history_frames),
        ]
        return np.concatenate(parts).astype(np.float32)

    def token_state_observation(self, frame, loop):
        if self.encoder is None:
            return self.token_state.copy()
        self.token_state = self.encoder(self.encoder_observation(frame, loop)).astype(np.float64)
        return self.token_state.copy()

    def encoder_observation(self, frame, loop):
        if self.encoder_input_dim == 1762:
            return self.encoder_observation_legacy1762(frame, loop)
        encoder_index = np.array([0.0], dtype=np.float64)
        command = self.motion_joint_pos_vel(frame, loop, self.history_frames, self.future_step)
        anchor = self.motion_anchor_orientation(frame, loop, self.history_frames, self.future_step)
        return np.concatenate([encoder_index, command, anchor]).astype(np.float32)

    def encoder_observation_legacy1762(self, frame, loop):
        obs = np.zeros(1762, dtype=np.float32)
        obs[0:4] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs[4:294] = self.motion_joint_positions(frame, loop, 10, 5).astype(np.float32)
        obs[294:584] = self.motion_joint_velocities(frame, loop, 10, 5).astype(np.float32)
        obs[601:661] = self.motion_anchor_orientation(frame, loop, 10, 5).astype(np.float32)
        return obs

    def history_values(self, attr, frames):
        if not self.history:
            raise RuntimeError("No robot snapshot in history")
        values = []
        hist = list(self.history)
        missing = max(0, frames - len(hist))
        if missing:
            values.extend(self.zero_history_value(attr) for _ in range(missing))
        for src in hist[-frames:]:
            values.append(np.asarray(getattr(src, attr), dtype=np.float64).reshape(-1))
        return np.concatenate(values)

    @staticmethod
    def zero_history_value(attr):
        if attr == "base_ang_vel":
            return np.zeros(3, dtype=np.float64)
        if attr in ("body_q_isaac", "body_dq_isaac", "last_action_isaac"):
            return np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        if attr == "gravity_dir":
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        raise ValueError(f"Unsupported history attr: {attr}")

    def motion_joint_pos_vel(self, frame, loop, frames, step):
        rows = []
        for i in range(frames):
            idx = self.reference.frame_index(frame + i * step, loop)
            rows.append(self.reference.joint_pos_isaac[idx])
            rows.append(self.reference.joint_vel_isaac[idx])
        return np.concatenate(rows)

    def motion_joint_positions(self, frame, loop, frames, step, joint_indexes=None):
        rows = []
        for i in range(frames):
            idx = self.reference.frame_index(frame + i * step, loop)
            value = self.reference.joint_pos_isaac[idx]
            rows.append(value if joint_indexes is None else value[joint_indexes])
        return np.concatenate(rows)

    def motion_joint_velocities(self, frame, loop, frames, step, joint_indexes=None):
        rows = []
        for i in range(frames):
            idx = self.reference.frame_index(frame + i * step, loop)
            value = self.reference.joint_vel_isaac[idx]
            rows.append(value if joint_indexes is None else value[joint_indexes])
        return np.concatenate(rows)

    def motion_anchor_orientation(self, frame, loop, frames, step):
        base = self.history[-1].base_quat if self.history else np.array([1.0, 0.0, 0.0, 0.0])
        init_base = self.init_base_quat if self.init_base_quat is not None else base
        apply_delta_heading = quat_mul(heading_quat(init_base), heading_quat_inv(self.init_ref_quat))
        rows = []
        for i in range(frames):
            idx = self.reference.frame_index(frame + i * step, loop)
            ref_quat = quat_mul(apply_delta_heading, self.reference.root_quat[idx])
            rows.append(rot6_from_quat_delta(base, ref_quat))
        return np.concatenate(rows)


class PolicyRunner:
    """策略推理与轨迹烘焙。"""

    def __init__(
        self,
        xml_path,
        motion_data_path,
        encoder_path,
        policy_path,
        motion_name=None,
        sim_dt=None,
    ):
        self.sim = MujocoG1Sim(xml_path, sim_dt)
        self.reference = load_reference_motion(motion_data_path, motion_name, "isaaclab")

        self.encoder = OrtRunner(encoder_path)
        self.policy = OrtRunner(policy_path)
        print(f"Loaded encoder {encoder_path}, input_shape={self.encoder.input_shape}")
        print(f"Loaded policy {policy_path}, input_shape={self.policy.input_shape}")

        self.obs_builder = ObservationBuilder(
            reference=self.reference,
            encoder=self.encoder,
            encoder_input_dim=self.encoder.expected_dim,
        )
        self.xml_path = xml_path

        self.sim_steps_per_control = max(1, round(CONTROL_DT / self.sim.model.opt.timestep))
        self.last_action = np.zeros(G1_NUM_MOTOR, dtype=np.float64)

        self._reset()

    def _reset(self):
        """重置仿真状态"""
        self.sim.reset_to_default_pose(0.793)
        self.obs_builder.reset_episode()
        self.last_action = np.zeros(G1_NUM_MOTOR, dtype=np.float64)

    def _policy_target_from_action(self, action_isaac):
        """将策略输出转换为 MuJoCo 关节目标位置"""
        if action_isaac.size != G1_NUM_MOTOR:
            raise RuntimeError(f"Policy action must be 29D, got {action_isaac.size}")
        return DEFAULT_ANGLES + action_isaac[ISAACLAB_TO_MUJOCO] * ACTION_SCALE

    def _step(self, frame, loop):
        """执行一个控制周期"""
        snapshot = RobotSnapshot(
            base_quat=self.sim.base_quat(),
            base_ang_vel=self.sim.base_ang_vel(),
            q_mujoco=self.sim.q_mujoco(),
            dq_mujoco=self.sim.dq_mujoco(),
            last_action_isaac=self.last_action.copy(),
        )
        self.obs_builder.push_snapshot(snapshot)

        obs = self.obs_builder.policy_observation(frame, loop)
        action_isaac = self.policy(obs).astype(np.float64)

        q_target = self._policy_target_from_action(action_isaac)
        dq_target = np.zeros(G1_NUM_MOTOR, dtype=np.float64)

        for _ in range(self.sim_steps_per_control):
            self.sim.apply_pd(q_target, dq_target, 1.0, 1.0)
            self.sim.step(1)

        self.last_action = action_isaac
        return action_isaac

    def bake_trajectory(self, max_steps=0, print_every=0):
        """烘焙完整轨迹"""
        print(f"Baking trajectory: {self.reference.name}, frames={self.reference.frames}")

        self._reset()
        max_frames = max_steps if max_steps > 0 else self.reference.frames

        qpos_traj = []
        qvel_traj = []
        frame = 0

        for _ in range(max_frames):
            self._step(frame, loop=False)
            qpos_traj.append(self.sim.data.qpos.copy())
            qvel_traj.append(self.sim.data.qvel.copy())

            if print_every > 0 and frame % print_every == 0:
                q_ref = self.reference.q_mujoco(frame, True)
                err = self.sim.q_mujoco() - q_ref
                print(
                    f"frame={frame:05d} "
                    f"joint_rmse={np.sqrt(np.mean(err**2)):.4f} "
                    f"joint_max={np.max(np.abs(err)):.4f}"
                )
            frame += 1

        print(f"Baked {len(qpos_traj)} frames")
        return np.stack(qpos_traj), np.stack(qvel_traj)

    def save_rollout(self, path, qpos_traj, qvel_traj):
        """保存烘焙轨迹到 .npz 文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            qpos=qpos_traj,
            qvel=qvel_traj,
            motion_name=np.array(self.reference.name),
            control_dt=np.array(CONTROL_DT),
            sim_dt=np.array(self.sim.model.opt.timestep),
            xml=np.array(str(self.xml_path)),
        )
        print(f"Saved rollout to {path}")