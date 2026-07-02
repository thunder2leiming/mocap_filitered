"""
动捕数据可视化与回放。
"""

import time
from pathlib import Path

import numpy as np

from config import CONTROL_DT, G1_NUM_MOTOR
from motion_data import load_reference_motion, ReferenceMotion
from simulator import MujocoG1Sim


class MotionVisualizer:
    """
    动捕数据可视化器。
    支持 kinematic 模式、pd 控制模式和缓存轨迹回放。
    """

    def __init__(self, xml_path, motion_data_path, motion_name=None, joint_order="isaaclab", sim_dt=None):
        """
        初始化可视化器。

        Args:
            xml_path: MuJoCo 模型 XML 文件路径
            motion_data_path: 参考动作数据目录
            motion_name: 动作名称 (可选)
            joint_order: 关节顺序 ("isaaclab" 或 "mujoco")
            sim_dt: 物理仿真步长 (可选)
        """
        self.sim = MujocoG1Sim(xml_path, sim_dt)
        self.reference = load_reference_motion(motion_data_path, motion_name, joint_order)
        self.xml_path = xml_path
        self.cached_qpos = None
        self.cached_qvel = None
        self.cached_motion_name = None

    def load_cached_rollout(self, npz_path):
        """
        加载已烘焙的轨迹缓存。

        Args:
            npz_path: .npz 文件路径
        """
        rollout = np.load(npz_path)
        self.cached_qpos = np.asarray(rollout["qpos"], dtype=np.float64)
        self.cached_qvel = np.asarray(rollout["qvel"], dtype=np.float64)
        self.cached_motion_name = (
            str(np.asarray(rollout["motion_name"]).item())
            if "motion_name" in rollout
            else npz_path.name
        )

        # 验证形状
        if self.cached_qpos.ndim != 2 or self.cached_qpos.shape[1] != self.sim.model.nq:
            raise ValueError(f"qpos shape {self.cached_qpos.shape} does not match model.nq={self.sim.model.nq}")
        if self.cached_qvel.ndim != 2 or self.cached_qvel.shape[1] != self.sim.model.nv:
            raise ValueError(f"qvel shape {self.cached_qvel.shape} does not match model.nv={self.sim.model.nv}")

        print(f"Loaded cached rollout: {self.cached_motion_name}, frames={self.cached_qpos.shape[0]}")

    def play_kinematic(self, loop=True, realtime=True, max_steps=0):
        """
        纯运动学播放（无物理仿真）。
        直接设置关节位置和速度到参考轨迹。

        Args:
            loop: 是否循环播放
            realtime: 是否实时同步
            max_steps: 最大步数 (0 表示完整动作)
        """
        print(f"Playing kinematic: {self.reference.name}, frames={self.reference.frames}, loop={loop}")

        try:
            import mujoco.viewer
        except ImportError:
            raise RuntimeError("Install mujoco first: pip install mujoco")

        max_frames = max_steps if max_steps > 0 else self.reference.frames
        frame = 0

        with mujoco.viewer.launch_passive(self.sim.model, self.sim.data) as viewer:
            while viewer.is_running():
                if frame >= max_frames:
                    if not loop:
                        break
                    frame = 0

                tick = time.perf_counter()
                self.sim.set_kinematic_reference(self.reference, frame, loop)
                viewer.sync()
                frame += 1

                if realtime:
                    sleep_s = CONTROL_DT - (time.perf_counter() - tick)
                    if sleep_s > 0:
                        time.sleep(sleep_s)

    def play_pd_control(self, loop=True, realtime=True, max_steps=0, kp_scale=1.0, kd_scale=1.0, pin_root=False):
        """
        PD 控制回放。
        使用参考关节角作为 PD 控制器目标。

        Args:
            loop: 是否循环播放
            realtime: 是否实时同步
            max_steps: 最大步数 (0 表示完整动作)
            kp_scale: PD 刚度缩放因子
            kd_scale: PD 阻尼缩放因子
            pin_root: 是否固定基座到参考轨迹
        """
        print(f"Playing PD control: {self.reference.name}, frames={self.reference.frames}, loop={loop}")

        try:
            import mujoco.viewer
        except ImportError:
            raise RuntimeError("Install mujoco first: pip install mujoco")

        self.sim.reset_to_reference(self.reference)
        max_frames = max_steps if max_steps > 0 else self.reference.frames
        sim_steps_per_control = max(1, round(CONTROL_DT / self.sim.model.opt.timestep))
        frame = 0

        with mujoco.viewer.launch_passive(self.sim.model, self.sim.data) as viewer:
            while viewer.is_running():
                if frame >= max_frames:
                    if not loop:
                        break
                    frame = 0
                    self.sim.reset_to_reference(self.reference)

                tick = time.perf_counter()

                # 获取参考目标
                q_target = self.reference.q_mujoco(frame, loop)
                dq_target = self.reference.dq_mujoco(frame, loop)

                # 执行物理子步
                for _ in range(sim_steps_per_control):
                    if pin_root:
                        self.sim.pin_root_to_reference(self.reference, frame, loop)
                    self.sim.apply_pd(q_target, dq_target, kp_scale, kd_scale)
                    self.sim.step(1)

                self.sim.forward()
                viewer.sync()
                frame += 1

                if realtime:
                    sleep_s = CONTROL_DT - (time.perf_counter() - tick)
                    if sleep_s > 0:
                        time.sleep(sleep_s)

    def play_cached(self, loop=True, realtime=True):
        """
        回放缓存的轨迹（不运行 ONNX）。

        Args:
            loop: 是否循环播放
            realtime: 是否实时同步
        """
        if self.cached_qpos is None:
            raise RuntimeError("No cached rollout loaded. Call load_cached_rollout() first.")

        print(f"Playing cached: {self.cached_motion_name}, frames={self.cached_qpos.shape[0]}, loop={loop}")

        try:
            import mujoco.viewer
        except ImportError:
            raise RuntimeError("Install mujoco first: pip install mujoco")

        frame = 0

        with mujoco.viewer.launch_passive(self.sim.model, self.sim.data) as viewer:
            while viewer.is_running():
                if frame >= self.cached_qpos.shape[0]:
                    if not loop:
                        break
                    frame = 0

                tick = time.perf_counter()
                # 直接设置状态
                self.sim.data.qpos[:] = self.cached_qpos[frame]
                self.sim.data.qvel[:] = self.cached_qvel[frame]
                self.sim.data.ctrl[:G1_NUM_MOTOR] = 0.0
                self.sim.forward()
                viewer.sync()
                frame += 1

                if realtime:
                    sleep_s = CONTROL_DT - (time.perf_counter() - tick)
                    if sleep_s > 0:
                        time.sleep(sleep_s)