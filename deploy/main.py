"""
简化版 SONIC 部署脚本。

运行模式:
  kinematic: 纯运动学播放参考轨迹
  pd:        PD 控制播放参考轨迹（带物理仿真）
  playback:  回放缓存的烘焙轨迹
  bake:      烘焙轨迹并可选回放

支持输入格式:
  - 标准 reference 目录 (包含 joint_pos.csv, joint_vel.csv 等)
  - SOMA CSV 文件 (单文件，包含 Frame, root_translate*, root_rotate*, *dof 列)
"""

import argparse
from pathlib import Path

from config import DEPLOY_ROOT, SIM_DT
from motion_visualizer import MotionVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # 模型与数据路径
    parser.add_argument("--xml", type=Path, default=DEPLOY_ROOT / "g1" / "scene_29dof.xml")
    parser.add_argument("--motion-data", type=Path, default=DEPLOY_ROOT / "reference" / "example")
    parser.add_argument("--motion-name", default="")
    
    # SOMA CSV 文件输入支持
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="输入 SOMA CSV 文件路径（例如: reference/dance_hiphop_shuffle_square_R_fast_002__A318.CSV）",
    )

    # 运行模式
    parser.add_argument(
        "--mode",
        choices=["kinematic", "pd", "playback", "bake"],
        default="kinematic",
        help="kinematic: 纯运动学播放; pd: PD控制播放; playback: 回放缓存; bake: 烘焙轨迹",
    )

    # PD 控制参数
    parser.add_argument("--kp-scale", type=float, default=1.0, help="PD刚度缩放因子")
    parser.add_argument("--kd-scale", type=float, default=1.0, help="PD阻尼缩放因子")
    parser.add_argument("--pin-root", action="store_true", help="固定基座到参考轨迹")

    # 仿真参数
    parser.add_argument("--sim-dt", type=float, default=SIM_DT)

    # 播放参数
    parser.add_argument("--loop", action="store_true", help="循环播放")
    parser.add_argument("--no-realtime", dest="realtime", action="store_false")
    parser.set_defaults(realtime=True)
    parser.add_argument("--max-steps", type=int, default=0, help="最大步数，0 表示完整动作")

    # 烘焙与回放
    parser.add_argument("--load-rollout", type=Path, default=None, help="加载缓存轨迹回放")
    parser.add_argument("--save-rollout", type=Path, default=None, help="保存烘焙轨迹")

    args = parser.parse_args()

    # 处理路径
    args.motion_name = args.motion_name or None
    args.xml = args.xml.resolve()
    
    # 处理 CSV 输入：如果指定了 --csv，使用 CSV 文件；否则使用 motion-data 目录
    if args.csv is not None:
        args.csv = args.csv.resolve()
        if not args.csv.exists():
            raise FileNotFoundError(f"CSV file not found: {args.csv}")
        # 如果提供了 CSV 文件，将其作为 motion-data
        args.motion_data = args.csv
    else:
        args.motion_data = args.motion_data.resolve()
    
    if args.save_rollout is not None:
        args.save_rollout = args.save_rollout.resolve()
    if args.load_rollout is not None:
        args.load_rollout = args.load_rollout.resolve()

    # 如果指定了 load-rollout，自动切换到 playback 模式
    if args.load_rollout is not None:
        args.mode = "playback"

    return args


def main():
    args = parse_args()

    # === playback 模式: 回放缓存轨迹 ===
    if args.mode == "playback":
        if args.load_rollout is None:
            raise ValueError("playback mode requires --load-rollout")

        visualizer = MotionVisualizer(
            xml_path=args.xml,
            motion_data_path=args.motion_data,
            motion_name=args.motion_name,
            sim_dt=args.sim_dt,
        )
        visualizer.load_cached_rollout(args.load_rollout)
        visualizer.play_cached(loop=args.loop, realtime=args.realtime)
        return

    # === kinematic 模式: 纯运动学播放 ===
    if args.mode == "kinematic":
        visualizer = MotionVisualizer(
            xml_path=args.xml,
            motion_data_path=args.motion_data,
            motion_name=args.motion_name,
            sim_dt=args.sim_dt,
        )
        visualizer.play_kinematic(
            loop=args.loop,
            realtime=args.realtime,
            max_steps=args.max_steps,
        )
        return

    # === pd 模式: PD 控制播放参考轨迹 ===
    if args.mode == "pd":
        visualizer = MotionVisualizer(
            xml_path=args.xml,
            motion_data_path=args.motion_data,
            motion_name=args.motion_name,
            sim_dt=args.sim_dt,
        )
        visualizer.play_pd_control(
            loop=args.loop,
            realtime=args.realtime,
            max_steps=args.max_steps,
            kp_scale=args.kp_scale,
            kd_scale=args.kd_scale,
            pin_root=args.pin_root,
        )
        return

    # === bake 模式: 烘焙轨迹 ===
    if args.mode == "bake":
        from policy_runner import PolicyRunner

        runner = PolicyRunner(
            xml_path=args.xml,
            motion_data_path=args.motion_data,
            encoder_path=DEPLOY_ROOT / "policy" / "release" / "model_encoder.onnx",
            policy_path=DEPLOY_ROOT / "policy" / "release" / "model_decoder.onnx",
            motion_name=args.motion_name,
            sim_dt=args.sim_dt,
        )

        qpos_traj, qvel_traj = runner.bake_trajectory(
            max_steps=args.max_steps,
            print_every=50,
        )

        if args.save_rollout is not None:
            runner.save_rollout(args.save_rollout, qpos_traj, qvel_traj)

        # 烘焙后回放
        visualizer = MotionVisualizer(
            xml_path=args.xml,
            motion_data_path=args.motion_data,
            motion_name=args.motion_name,
            sim_dt=args.sim_dt,
        )
        visualizer.cached_qpos = qpos_traj
        visualizer.cached_qvel = qvel_traj
        visualizer.cached_motion_name = runner.reference.name
        visualizer.play_cached(loop=args.loop, realtime=args.realtime)
        return


if __name__ == "__main__":
    main()