"""
MuJoCo G1 仿真封装。
"""

from pathlib import Path

import numpy as np

from config import (
    DEFAULT_ANGLES,
    G1_NUM_MOTOR,
    KDS,
    KPS,
    SIM_DT,
    TORQUE_LIMITS,
)
from motion_data import ReferenceMotion


class MujocoG1Sim:
    """
    MuJoCo G1 仿真封装。
    管理模型加载、状态读写、PD 控制和物理步进。
    """
    def __init__(self, xml_path, sim_dt=SIM_DT):
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
        self.gyro_sensor_adr = None
        self.gyro_sensor_dim = 0
        sensor_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_SENSOR, "imu-angular-velocity"
        )
        if sensor_id >= 0:
            self.gyro_sensor_adr = int(self.model.sensor_adr[sensor_id])
            self.gyro_sensor_dim = int(self.model.sensor_dim[sensor_id])

    def reset_to_reference(self, reference):
        """重置仿真到参考动作的第一帧"""
        self.mujoco.mj_resetData(self.model, self.data)
        self.set_kinematic_reference(reference, 0, loop=False)

    def reset_to_default_pose(self, root_height=0.793):
        """重置到默认站立姿态 (策略启动常用)"""
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = np.array([0.0, 0.0, root_height], dtype=np.float64)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.set_joint_qpos(DEFAULT_ANGLES)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:G1_NUM_MOTOR] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def set_joint_qpos(self, q):
        """设置所有电机关节位置"""
        self.data.qpos[self.qpos_addrs] = q

    def set_joint_qvel(self, dq):
        """设置所有电机关节速度"""
        self.data.qvel[self.qvel_addrs] = dq

    def q_mujoco(self):
        """读取当前关节位置"""
        return self.data.qpos[self.qpos_addrs].copy()

    def dq_mujoco(self):
        """读取当前关节速度"""
        return self.data.qvel[self.qvel_addrs].copy()

    def base_quat(self):
        """读取基座四元数 (free joint 的前 4 个 qpos 分量)"""
        return self.data.qpos[3:7].copy()

    def base_ang_vel(self):
        """
        读取基座角速度。
        优先使用 IMU 陀螺仪传感器数据（更接近真实部署），
        回退到 free joint 的 qvel。
        """
        if self.gyro_sensor_adr is not None and self.gyro_sensor_dim == 3:
            adr = self.gyro_sensor_adr
            return self.data.sensordata[adr : adr + 3].copy()
        return self.data.qvel[3:6].copy()

    def set_root_pose(self, reference, frame, loop):
        """设置基座位姿到参考动作指定帧"""
        idx = reference.frame_index(frame, loop)
        self.data.qpos[0:3] = reference.root_pos[idx]
        self.data.qpos[3:7] = reference.root_quat[idx]

    def pin_root_to_reference(self, reference, frame, loop):
        """
        将基座强制固定到参考动作位姿 (诊断用)。
        同时清零基座速度，实现完全运动学约束。
        """
        self.set_root_pose(reference, frame, loop)
        self.data.qvel[0:6] = 0.0

    def set_kinematic_reference(self, reference, frame, loop):
        """完全运动学设置 (位置+速度+基座)，用于 kinematic 模式"""
        self.set_root_pose(reference, frame, loop)
        self.set_joint_qpos(reference.q_mujoco(frame, loop))
        self.set_joint_qvel(reference.dq_mujoco(frame, loop))
        self.data.ctrl[:G1_NUM_MOTOR] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def apply_pd(self, q_target, dq_target, kp_scale=1.0, kd_scale=1.0):
        """
        应用 PD 控制器计算力矩并写入 ctrl。
        tau = Kp * scale * (q_target - q) + Kd * scale * (dq_target - dq)
        结果裁剪到力矩限制范围内。
        """
        q = self.q_mujoco()
        dq = self.dq_mujoco()
        tau = (KPS * kp_scale) * (q_target - q) + (KDS * kd_scale) * (dq_target - dq)
        self.data.ctrl[:G1_NUM_MOTOR] = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)

    def step(self, n=1):
        """执行 n 步物理仿真"""
        for _ in range(n):
            self.mujoco.mj_step(self.model, self.data)

    def forward(self):
        """执行 mj_forward 更新派生状态"""
        self.mujoco.mj_forward(self.model, self.data)