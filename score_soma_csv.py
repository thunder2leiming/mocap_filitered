"""
SOMA CSV格式运动数据物理评分脚本

功能：
1. 读取SOMA CSV格式数据
2. 转换为qpos格式（36维：root_pos(3) + root_quat(4) + joints(29)）
3. 使用physics_filter.py的逻辑进行物理可行性评分
4. 输出评分结果JSON

数据格式要求：
- 输入：SOMA CSV (root_translateX/Y/Z, root_rotateX/Y/Z, 29个关节dof)
- 输出：评分JSON + 过滤后的.npz文件
"""

import gc
import os
import sys
import tyro
import time
import json
import math
import shutil
import signal
from pathlib import Path
import multiprocessing as mp
from functools import partial
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# 内联常量（原来自 tracking/constants.py）
# 机器人模型XML配置文件路径
TRACK_XML = Path("storage/assets/unitree_g1_5010/scene_mjx_track.xml")

# 各关节速度限制值列表
DOF_VEL_LIMITS = [
    32.0, 32.0, 32.0, 20.0, 37.0, 37.0,  # 前6个关节速度限制
    32.0, 32.0, 32.0, 20.0, 37.0, 37.0,  # 中间12个关节速度限制
    32.0, 37.0, 37.0,                     # 髋部关节速度限制
    37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0,  # 膝部关节速度限制
    37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0   # 踝部关节速度限制
]


def count_contacts_exclude(
    data: mjx.Data, geom_id_exclude: int, only_colliding: bool = True
) -> jax.Array:
    """
    计算不包含排除几何体的接触点数量
    
    Args:
        data: mujoco mjx数据对象
        geom_id_exclude: 需要排除的几何体ID（通常是地面）
        only_colliding: 是否仅计算碰撞状态的接触点
    
    Returns:
        接触点数量数组
    """
    geom_pairs = data.contact.geom  # [N, 2] int 接触对数组
    not_target = (geom_pairs != geom_id_exclude).all(axis=1)  # 排除指定几何体的接触对
    n = geom_pairs.shape[0]
    active = jnp.arange(n) < data.ncon  # 活跃接触点掩码

    mask = not_target & active  # 有效接触点掩码
    if only_colliding:
        mask = mask & (data.contact.dist < 0)  # 只考虑实际碰撞的接触点

    return jnp.sum(mask)


class PhysicsScoringEnv:
    """物理评分环境类 - 最小化的MJX模型容器用于物理可行性评分"""

    def __init__(self, xml_path: str | None = None):
        # 使用内联常量或自定义路径加载机器人模型
        xml_file = str(xml_path) if xml_path is not None else str(TRACK_XML)
        mj_model = mujoco.MjModel.from_xml_path(xml_file)
        self.mjx_model = mjx.put_model(mj_model)  # 转换为JAX格式模型
        self.dof_vel_limit = jnp.array(DOF_VEL_LIMITS)  # 关节速度限制
        
        # 获取关键身体部位ID
        self.body_id_ankle_l = mj_model.body("left_ankle_roll_link").id  # 左踝关节ID
        self.body_id_ankle_r = mj_model.body("right_ankle_roll_link").id  # 右踝关节ID
        self.geom_id_floor = mj_model.geom("floor").id  # 地面几何体ID


def convert_soma_csv_to_qpos(csv_path: str, freq: float = 30.0) -> Optional[Dict[str, np.ndarray]]:
    """
    将SOMA CSV格式转换为qpos/qvel格式
    
    Args:
        csv_path: SOMA CSV文件路径
        freq: 采样频率（默认30Hz）
    
    Returns:
        包含qpos、qvel和频率的字典，如果转换失败则返回None
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"读取文件错误 {csv_path}: {e}")
        return None
    
    # 1. 提取根位置（厘米转米）
    root_pos = df[['root_translateX', 'root_translateY', 'root_translateZ']].to_numpy() * 0.01
    
    # 2. 根旋转：欧拉角XYZ（度）→ 四元数(x, y, z, w)
    euler_xyz_deg = df[['root_rotateX', 'root_rotateY', 'root_rotateZ']].to_numpy()
    rot = R.from_euler('XYZ', euler_xyz_deg, degrees=True)
    quat_xyzw = rot.as_quat()  # scipy返回的是(x,y,z,w)格式
    
    # 3. 提取关节角度（度转弧度）
    joint_cols = [c for c in df.columns if c.endswith('_dof')]
    if len(joint_cols) != 29:
        print(f"警告: 期望29个关节，实际得到 {len(joint_cols)} 个在 {csv_path}")
        return None
    
    joint_angles = np.deg2rad(df[joint_cols].to_numpy())
    
    # 4. 拼接qpos: [pos(3), quat(4), joints(29)] = 36维
    qpos = np.hstack([root_pos, quat_xyzw, joint_angles]).astype(np.float32)
    
    # 5. 计算qvel（数值微分）
    dt = 1.0 / freq
    qvel = np.zeros((len(qpos), 35), dtype=np.float32)
    
    # 根速度（位置差分）
    if len(qpos) > 1:
        qvel[:-1, :3] = np.diff(qpos[:, :3], axis=0) / dt
        qvel[-1:, :3] = qvel[-2:-1, :3]  # 最后一帧复制
        
        # 根角速度（四元数差分）
        # 简化处理：使用欧拉角差分
        euler_prev = euler_xyz_deg[:-1]
        euler_curr = euler_xyz_deg[1:]
        euler_vel = (euler_curr - euler_prev) / dt
        qvel[:-1, 3:6] = np.deg2rad(euler_vel)
        qvel[-1:, 3:6] = qvel[-2:-1, 3:6]
        
        # 关节速度
        qvel[:-1, 6:] = np.diff(joint_angles, axis=0) / dt
        qvel[-1:, 6:] = qvel[-2:-1, 6:]

    return {
        'qpos': qpos,
        'qvel': qvel,
        'frequency': freq
    }


@dataclass
class Args:
    """命令行参数配置类"""
    # 输入输出路径
    csv_dir: str = "csv/amass2soma"  # 输入CSV目录
    score_json_path: str = "storage/gqs_score/soma_csv_scores.json"  # 评分结果JSON路径
    output_csv_dir: str = "storage/mocap/soma_converted"  # 输出CSV目录
    output_filtered_dir: str = "storage/mocap/soma_filtered"  # 过滤后输出目录
    
    # 评分参数
    dt: float = 0.0333  # 时间步长 (30Hz → 0.0333s)
    source_freq: float = 30.0  # 源数据频率
    num_gpus: int = -1  # GPU数量 (-1表示自动检测)
    min_duration: float = 0.5  # 最小动作持续时间阈值(秒)
    
    # 过滤参数
    threshold: float = 90.0  # 分数阈值


# 全局变量
_processes: Optional[List[mp.Process]] = None
_queue: Optional[mp.Queue] = None
_interrupted: bool = False


def signal_handler(signum, frame):
    """信号处理器 - 优雅地终止子进程"""
    global _processes, _queue, _interrupted
    if _interrupted:
        sys.exit(1)
    _interrupted = True
    print("\n停止子进程...")
    if _processes:
        for p in _processes:
            if p.is_alive(): p.terminate()
        for p in _processes:
            p.join(timeout=2)
            if p.is_alive(): p.kill()
    print("已停止.")
    raise KeyboardInterrupt


def compute_single_frame(
    sys: mjx.Model,
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    prev_qvel: jnp.ndarray,
    body_id_l: int,
    body_id_r: int,
    geom_id_floor: int,
    dof_vel_limit: jnp.ndarray,
    dt: float
) -> Tuple[float, float, float, float, float, float]:
    """
    计算单帧物理评分指标
    
    Args:
        sys: mujoco模型
        qpos: 关节位置
        qvel: 关节速度
        prev_qvel: 上一帧关节速度
        body_id_l: 左踝关节ID
        body_id_r: 右踝关节ID
        geom_id_floor: 地面几何体ID
        dof_vel_limit: 关节速度限制
        dt: 时间步长
    
    Returns:
        (p_slide, p_vel, p_col, p_jerk, p_pen, is_air) - 各项惩罚值
    """

    d = mjx.make_data(sys)
    d = d.replace(qpos=qpos, qvel=qvel)
    # mjx.forward执行碰撞检测并填充d.contact
    d = mjx.forward(sys, d)

    # --- 1. 足部滑动 ---
    left_vel = d.cvel[body_id_l][3:5]  # 左踝线速度xy分量
    right_vel = d.cvel[body_id_r][3:5]  # 右踝线速度xy分量
    l_speed = jnp.linalg.norm(left_vel)
    r_speed = jnp.linalg.norm(right_vel)

    # 简单足部接触检测用于滑动计算
    l_contact = d.xpos[body_id_l][2] < 0.05  # Z坐标小于0.05认为接触地面
    r_contact = d.xpos[body_id_r][2] < 0.05

    p_slide = 0.0
    # 如果接触地面且速度大于0.1，则施加惩罚
    p_slide += jnp.where(l_contact, jnp.maximum(0.0, l_speed - 0.1), 0.0)
    p_slide += jnp.where(r_contact, jnp.maximum(0.0, r_speed - 0.1), 0.0)
    p_slide *= 5.0  # 滑动惩罚放大系数

    # --- 2. 速度限制 ---
    joint_vels = qvel[-len(dof_vel_limit):]  # 取关节速度部分
    p_vel = jnp.mean(jnp.maximum(0.0, jnp.abs(joint_vels) - dof_vel_limit))

    # --- 3. 自碰撞 ---
    n_con = count_contacts_exclude(d, geom_id_floor, only_colliding=True)
    p_col = jnp.clip(n_con, 0.0, 10.0)  # 限制最大碰撞惩罚值

    # --- 4. 急动度(Jerk) ---
    accel = (qvel - prev_qvel) / dt  # 加速度
    p_jerk = jnp.linalg.norm(accel) * 0.01  # 急动度惩罚系数

    # --- 5 & 6. 全局接触分析(任意身体部位) ---
    floor_mask = (d.contact.geom1 == geom_id_floor) | (d.contact.geom2 == geom_id_floor)
    dists_to_floor = jnp.where(floor_mask, d.contact.dist, 100.0)
    min_dist = jnp.min(dists_to_floor)

    # [穿透检测]
    p_pen = jnp.maximum(0.0, -min_dist - 0.01)  # 穿透惩罚

    # [悬空检测]
    is_air = jnp.where(min_dist > 0.05, 1.0, 0.0)  # 高于0.05认为悬空

    return p_slide, p_vel, p_col, p_jerk, p_pen, is_air


@partial(jax.jit, static_argnums=(5, 6, 7, 9))
def compute_clip_metrics_jit(
    sys: mjx.Model,
    qpos_seq: jnp.ndarray,
    qvel_seq: jnp.ndarray,
    prev_qvel_seq: jnp.ndarray,
    mask: jnp.ndarray,
    body_id_l: int,
    body_id_r: int,
    geom_id_floor: int,
    dof_vel_limit: jnp.ndarray,
    dt: float,
):
    """
    JAX JIT编译的批量计算函数
    
    Args:
        sys: mujoco模型
        qpos_seq: 关节位置序列
        qvel_seq: 关节速度序列
        prev_qvel_seq: 上一帧速度序列
        mask: 有效帧掩码
        其他参数同compute_single_frame
    
    Returns:
        各项惩罚总和
    """
    vmap_fn = jax.vmap(
        compute_single_frame,
        in_axes=(None, 0, 0, 0, None, None, None, None, None)
    )

    s, v, c, j, p, air = vmap_fn(
        sys, qpos_seq, qvel_seq, prev_qvel_seq,
        body_id_l, body_id_r, geom_id_floor, dof_vel_limit, dt
    )

    mask = mask.astype(s.dtype)

    total_slide = jnp.sum(s * mask)
    total_vel = jnp.sum(v * mask)
    total_col = jnp.sum(c * mask)
    total_jerk = jnp.sum(j * mask)
    total_pen = jnp.sum(p * mask)

    # --- 长期悬空检测 ---
    valid_air = air * mask
    window_size = int(1.0 / dt)  # 1秒窗口
    kernel = jnp.ones(window_size)
    conv_res = jnp.convolve(valid_air, kernel, mode='same')

    floating_violation_frames = jnp.where(conv_res >= (window_size - 0.1), 1.0, 0.0)
    total_float_frames = jnp.sum(floating_violation_frames * mask)

    return total_slide, total_vel, total_col, total_jerk, total_pen, total_float_frames


def get_padded_batch(qpos, qvel, prev_qvel, chunk_size=512):
    """
    获取填充批次数据
    
    Args:
        qpos, qvel, prev_qvel: 关节位置、速度、上一帧速度
        chunk_size: 批次大小
    
    Returns:
        填充后的数据和掩码
    """
    n_frames = qpos.shape[0]
    if n_frames == 0: return None, None, None, None

    target_len = math.ceil(n_frames / chunk_size) * chunk_size
    pad_len = target_len - n_frames

    # 创建有效帧掩码
    mask = np.concatenate([np.ones(n_frames), np.zeros(pad_len)])
    # 边缘填充位置数据，常数填充速度数据
    qpos_pad = np.pad(qpos, ((0, pad_len), (0, 0)), mode='edge')
    qvel_pad = np.pad(qvel, ((0, pad_len), (0, 0)), mode='constant')
    prev_qvel_pad = np.pad(prev_qvel, ((0, pad_len), (0, 0)), mode='constant')

    return qpos_pad, qvel_pad, prev_qvel_pad, mask


def score_one_csv(csv_path: Path, env: PhysicsScoringEnv, args: Args) -> Tuple[float, Dict]:
    """
    评分单个CSV文件
    
    Args:
        csv_path: CSV文件路径
        env: 物理环境实例
        args: 参数配置
    
    Returns:
        (评分, 详细指标字典)
    """
    
    # 1. 转换CSV到qpos/qvel
    data = convert_soma_csv_to_qpos(str(csv_path), freq=args.source_freq)
    if data is None:
        return 0.0, {"error": "conversion_failed"}
    
    qpos_np = data["qpos"]
    qvel_np = data["qvel"]
    n_frames = len(qpos_np)
    
    # 2. 保存原始CSV到输出目录（后续过滤使用）
    output_csv_path = Path(args.output_csv_dir) / f"{csv_path.stem}.csv"
    shutil.copy2(str(csv_path), str(output_csv_path))
    
    # === [硬过滤1] 持续时间检查 ===
    duration = n_frames * args.dt
    if n_frames < 5 or duration < args.min_duration:
        zero_metrics = {
            "foot_sliding": 100.0, "velocity_violation": 100.0,
            "self_collision": 100.0, "jerk": 100.0,
            "penetration": 100.0, "floating_frames_ratio": 1.0,
            "is_too_short": 1.0
        }
        return 0.0, zero_metrics

    # 构建前一帧速度序列
    prev_qvel_seq_np = np.concatenate([qvel_np[:1], qvel_np[:-1]])

    qpos_pad, qvel_pad, prev_qvel_pad, mask = get_padded_batch(
        qpos_np, qvel_np, prev_qvel_seq_np, chunk_size=512
    )

    qpos_j = jnp.array(qpos_pad)
    qvel_j = jnp.array(qvel_pad)
    prev_qvel_j = jnp.array(prev_qvel_pad)
    mask_j = jnp.array(mask)
    vlim = env.dof_vel_limit

    # 执行批量化评分计算
    t_slide, t_vel, t_col, t_jerk, t_pen, t_float = compute_clip_metrics_jit(
        env.mjx_model,
        qpos_j, qvel_j, prev_qvel_j, mask_j,
        env.body_id_ankle_l, env.body_id_ankle_r, env.geom_id_floor,
        vlim, args.dt
    )

    # 计算各项指标平均值
    metrics = {
        "foot_sliding": float(t_slide) / n_frames,
        "velocity_violation": float(t_vel) / n_frames,
        "self_collision": float(t_col) / n_frames,
        "jerk": float(t_jerk) / n_frames,
        "penetration": float(t_pen) / n_frames,
        "floating_frames_ratio": float(t_float) / n_frames
    }

    # === [软评分] 权重配置 ===
    # 根据各项指标计算最终评分
    score = 100.0 - (
        1.0 * metrics["foot_sliding"] +           # 足部滑动 - 基础优先级
        5.0 * metrics["velocity_violation"] +     # 速度违规 - 高优先级(在有效范围内)
        10 * metrics["self_collision"] +          # 自碰撞 - 高优先级(在有效范围内)
        0.01 * metrics["jerk"] +                  # 急动度 - 忽略
        10.0 * metrics["penetration"] +           # 穿透 - 减少权重(允许软穿透)
        200.0 * metrics["floating_frames_ratio"]  # 悬空帧比例 - 最高优先级(双倍权重)
    )

    return max(0.0, score), metrics


def worker_process(gpu_id: int, file_subset: List[Path], args: Args, return_queue: mp.Queue, launch_delay: float = 0.0):
    """
    工作进程函数 - 在指定GPU上处理文件子集
    
    Args:
        gpu_id: GPU ID
        file_subset: 文件子集
        args: 参数配置
        return_queue: 返回队列
        launch_delay: 启动延迟
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    # 为每个工作进程创建隔离的临时目录
    worker_tmp = Path(os.environ.get('TMPDIR', '/tmp')) / f"physics_filter_gpu{gpu_id}_{os.getpid()}"
    worker_tmp.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(worker_tmp)

    # 错开启动时间
    if launch_delay > 0:
        time.sleep(launch_delay)

    print(f"[GPU {gpu_id}] 已启动. 文件数: {len(file_subset)}  临时目录={worker_tmp}", flush=True)

    file_names_remaining = [f.name for f in file_subset]

    try:
        env = PhysicsScoringEnv()

        # 预热JIT编译
        dummy_q = jnp.zeros((512, env.mjx_model.nq))
        dummy_v = jnp.zeros((512, env.mjx_model.nv))
        dummy_mask = jnp.ones(512)
        _ = compute_clip_metrics_jit(
            env.mjx_model, dummy_q, dummy_v, dummy_v, dummy_mask,
            env.body_id_ankle_l, env.body_id_ankle_r, env.geom_id_floor,
            env.dof_vel_limit, 0.0333
        )

        batch_size = 100
        for idx, f in enumerate(tqdm(file_subset, position=gpu_id, desc=f"GPU {gpu_id}")):
            try:
                score, mets = score_one_csv(f, env, args)
                return_queue.put(('result', f.name, score, mets))
                file_names_remaining.remove(f.name)
                if (idx + 1) % batch_size == 0:
                    return_queue.put(('batch', gpu_id, idx + 1, None))
                if (idx + 1) % 500 == 0: gc.collect()
            except Exception as e:
                return_queue.put(('error', f.name, None, str(e)))
                file_names_remaining.remove(f.name)

        return_queue.put(('done', gpu_id, [], None))
    except Exception as e:
        err_msg = f"[GPU {gpu_id}] 工作进程中致命错误: {type(e).__name__}: {e}"
        print(err_msg, flush=True)
        return_queue.put(('done', gpu_id, file_names_remaining, err_msg))


def load_existing_results(score_json_path: str):
    """
    加载已有评分结果
    
    Args:
        score_json_path: 评分JSON文件路径
    
    Returns:
        (摘要结果, 详细结果)
    """
    res, det = {}, {}
    path = Path(score_json_path)
    if path.exists():
        try:
            with open(path, 'r') as f: data = json.load(f)
            if "summary" in data: res = {k: v for k, v in data["summary"]}  # 摘要结果
            if "details" in data: det = data["details"]  # 详细结果
            print(f"已加载 {len(res)} 个现有结果.")
        except: pass
    return res, det


def save_results(path, res, det, new_res, new_det):
    """
    保存评分结果
    
    Args:
        path: 保存路径
        res: 现有结果
        det: 现有详细结果
        new_res: 新增结果
        new_det: 新增详细结果
    
    Returns:
        合并后的结果字典
    """
    final_res = {**res, **new_res}
    final_det = {**det, **new_det}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = {
        "summary": sorted(final_res.items(), key=lambda x: x[1], reverse=True),
        "details": final_det
    }
    tmp = path + ".tmp"
    with open(tmp, 'w') as f: json.dump(out, f, indent=2)
    # Windows兼容：只在文件存在时删除
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)
    return final_res


def run_scoring(args: Args) -> Dict[str, float]:
    """
    运行多GPU评分
    
    Args:
        args: 参数配置
    
    Returns:
        评分结果字典
    """
    print("=" * 60)
    print("阶段1: 基于物理的质量评分 (SOMA CSV)")
    print("=" * 60)

    existing_res, existing_det = load_existing_results(args.score_json_path)
    csv_dir = Path(args.csv_dir)
    all_files = sorted(list(csv_dir.rglob("*.csv")))
    files = [f for f in all_files if f.name not in existing_res]

    print(f"CSV文件总数: {len(all_files)}, 已评分: {len(existing_res)}, 剩余: {len(files)}")

    if not files:
        print("所有文件均已评分. 跳过评分阶段.")
        return existing_res

    if args.num_gpus == -1:
        try:
            import subprocess
            output = subprocess.check_output("nvidia-smi -L", shell=True).decode().strip()
            avail_gpus = len(output.split("\n"))
        except: avail_gpus = 1
    else: avail_gpus = args.num_gpus

    print(f"在 {avail_gpus} 个GPU上启动.")
    chunk_size = math.ceil(len(files) / avail_gpus)
    chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]

    global _processes, _queue
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    mp.set_start_method('spawn', force=True)
    queue = mp.Queue()
    processes = []
    _processes = processes
    _queue = queue

    new_res, new_det = {}, {}
    done_workers = set()
    failed_files: List[str] = []
    error_files: List[Tuple[str, str]] = []
    final_scores = existing_res.copy()

    name_to_path = {f.name: f for f in files}

    try:
        startup_stagger = 3.0
        for i in range(min(avail_gpus, len(chunks))):
            delay = i * startup_stagger
            p = mp.Process(target=worker_process, args=(i, chunks[i], args, queue, delay))
            p.start()
            processes.append(p)

        last_save = time.time()
        while len(done_workers) < len(processes):
            try:
                msg = queue.get(timeout=1.0)
                if msg[0] == 'result':
                    new_res[msg[1]] = msg[2]
                    new_det[msg[1]] = msg[3]
                elif msg[0] == 'batch':
                    if time.time() - last_save > 60:
                        save_results(args.score_json_path, existing_res, existing_det, new_res, new_det)
                        last_save = time.time()
                        print(f"已保存. 新增: {len(new_res)}")
                elif msg[0] == 'error':
                    error_files.append((msg[1], msg[3] if len(msg) > 3 else ''))
                elif msg[0] == 'done':
                    done_workers.add(msg[1])
                    remaining = msg[2] if len(msg) > 2 and msg[2] else []
                    if remaining:
                        failed_files.extend(remaining)
                        err_str = msg[3] if len(msg) > 3 and msg[3] else '未知错误'
                        print(f"[GPU {msg[1]}] 工作进程失败. {len(remaining)} 个文件未处理. ({err_str})")
            except:
                for i, p in enumerate(processes):
                    if not p.is_alive() and i not in done_workers:
                        done_workers.add(i)
                        print(f"[GPU {i}] 进程死亡但未发送'done'. 其块中的文件将被重试.")

        for p in processes: p.join()

        input_names = set(name_to_path.keys())
        accounted = set(new_res.keys()) | set(failed_files) | set(name for name, _ in error_files)
        silently_lost = input_names - accounted
        if silently_lost:
            print(f"警告: {len(silently_lost)} 个文件静默丢失(无结果, 无错误). 将重试.")
            failed_files.extend(silently_lost)

        final_scores = save_results(args.score_json_path, existing_res, existing_det, new_res, new_det)
        print(f"并行阶段完成. 目前评分: {len(final_scores)}, "
              f"失败/丢失: {len(failed_files)}, 单文件错误: {len(error_files)}")

        # 恢复阶段
        if failed_files:
            print("\n" + "=" * 60)
            print(f"恢复阶段: 在GPU 0上串行重新评分 {len(failed_files)} 个未处理文件")
            print("=" * 60)
            retry_queue = mp.Queue()
            retry_chunk = [name_to_path[n] for n in failed_files if n in name_to_path]
            rp = mp.Process(target=worker_process, args=(0, retry_chunk, args, retry_queue, 0.0))
            rp.start()
            retry_done = False
            while not retry_done:
                try:
                    msg = retry_queue.get(timeout=1.0)
                    if msg[0] == 'result':
                        new_res[msg[1]] = msg[2]
                        new_det[msg[1]] = msg[3]
                    elif msg[0] == 'done':
                        retry_done = True
                        remaining = msg[2] if len(msg) > 2 and msg[2] else []
                        if remaining:
                            print(f"恢复仍缺少 {len(remaining)} 个文件(工作进程也崩溃了).")
                except:
                    if not rp.is_alive():
                        retry_done = True
            rp.join()
            final_scores = save_results(args.score_json_path, existing_res, existing_det, new_res, new_det)
            print(f"恢复完成. 总评分: {len(final_scores)}")
        else:
            print(f"评分完成. 总评分: {len(final_scores)}")

    except KeyboardInterrupt:
        if new_res:
            final_scores = save_results(args.score_json_path, existing_res, existing_det, new_res, new_det)
        raise
    except Exception as e:
        print(f"错误: {e}")
        if new_res:
            final_scores = save_results(args.score_json_path, existing_res, existing_det, new_res, new_det)
    finally:
        _processes = None

    return final_scores


def run_filtering(args: Args, score_map: Dict[str, float]):
    """
    过滤并复制通过阈值的CSV文件
    
    Args:
        args: 参数配置
        score_map: 评分映射字典
    """
    print("\n" + "=" * 60)
    print("阶段2: 过滤和复制通过的动作")
    print("=" * 60)

    os.makedirs(args.output_filtered_dir, exist_ok=True)

    print(f"扫描转换后的CSV目录: {args.output_csv_dir}...")
    src_files = list(Path(args.output_csv_dir).rglob("*.csv"))

    print(f"找到 {len(src_files)} 个CSV文件.")
    print(f"使用阈值 >= {args.threshold} 进行过滤...")

    passed_count = 0
    already_copied_count = 0
    skipped_count = 0
    missing_score_count = 0

    for src_path in tqdm(src_files, desc="复制通过的动作"):
        file_name = src_path.name
        score = score_map.get(file_name)

        if score is None:
            missing_score_count += 1
            continue

        if score >= args.threshold:
            dst_path = os.path.join(args.output_filtered_dir, file_name)
            try:
                if os.path.exists(dst_path) and os.path.getsize(dst_path) == src_path.stat().st_size:
                    already_copied_count += 1
                else:
                    shutil.copy2(src_path, dst_path)
                    passed_count += 1
            except OSError:
                shutil.copy2(src_path, dst_path)
                passed_count += 1
        else:
            skipped_count += 1

    print("-" * 50)
    print(f"过滤完成.")
    print(f"  阈值:           {args.threshold}")
    print(f"  总源文件:        {len(src_files)}")
    print(f"  新复制:        {passed_count}")
    print(f"  已复制:      {already_copied_count}")
    print(f"  总通过:        {passed_count + already_copied_count}")
    print(f"  被拒绝:            {skipped_count}")
    print(f"  缺少评分:       {missing_score_count}")
    print(f"输出目录:      {args.output_filtered_dir}")
    print("-" * 50)


def main(args: Args):
    """
    主函数 - 执行完整的评分和过滤流水线
    
    Args:
        args: 参数配置
    """
    print("=" * 60)
    print("SOMA CSV基于物理的质量评分流水线")
    print("=" * 60)
    print(f"输入CSV目录:     {args.csv_dir}")
    print(f"评分JSON路径:         {args.score_json_path}")
    print(f"输出CSV目录:    {args.output_csv_dir}")
    print(f"输出过滤目录:     {args.output_filtered_dir}")
    print(f"源频率:        {args.source_freq} Hz")
    print(f"阈值:               {args.threshold}")
    print("=" * 60)

    # 创建输出目录
    Path(args.output_csv_dir).mkdir(parents=True, exist_ok=True)

    # 阶段1: 评分
    score_map = run_scoring(args)

    # 阶段2: 过滤
    run_filtering(args, score_map)

    print("\n" + "=" * 60)
    print("流水线完成!")
    print("=" * 60)


if __name__ == "__main__":
    main(tyro.cli(Args))