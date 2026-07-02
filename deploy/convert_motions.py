import sys
import os

def convert_motion_data(pkl_file, base_output_dir=None):
    """
    将运动pickle文件转换为C++可读格式
    
    参数:
    - pkl_file: 输入的pickle文件路径
    - base_output_dir: 输出基础目录路径，默认为None（使用参考文件夹结构）
    
    返回:
    - 转换结果统计信息
    """
    
    # 创建有序的文件夹结构: reference/{pkl_name}/
    # 提取文件名（不含扩展名）
    pkl_name = os.path.splitext(os.path.basename(pkl_file))[0]
    
    if base_output_dir is None:
        # 默认：放在参考文件夹结构中
        base_output_dir = os.path.join(os.path.dirname(pkl_file), pkl_name)
    
    print(f"正在转换运动数据: {pkl_file}")
    print(f"输出目录结构: {base_output_dir}/")
    
    # 加载数据
    try:
        # 首先尝试使用joblib加载
        import joblib
        data = joblib.load(pkl_file)
        print("✓ 成功使用joblib加载数据")
    except ImportError:
        # 如果joblib不可用，则尝试使用pickle
        print("✗ joblib不可用，尝试使用pickle...")
        try:
            import pickle
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            print("✓ 成功使用pickle加载数据")
        except Exception as e:
            print(f"✗ 加载失败: {e}")
            return False
    except Exception as e:
        print(f"✗ 使用joblib加载失败: {e}")
        return False
    
    # 创建基本输出目录
    os.makedirs(base_output_dir, exist_ok=True)
    
    print(f"\n找到 {len(data)} 个运动序列:")
    for motion_name in data.keys():
        print(f"  - {motion_name}")
    
    # 转换每个运动序列
    success_count = 0
    for motion_name, motion_data in data.items():
        print(f"\n正在处理: {motion_name}")
        
        # 为此运动创建单独的文件夹
        motion_output_dir = os.path.join(base_output_dir, motion_name)
        print(f"为此运动创建单独文件夹: {motion_output_dir}")
        os.makedirs(motion_output_dir, exist_ok=True)
        
        if convert_single_motion(motion_name, motion_data, motion_output_dir):
            success_count += 1
    
    # 在基础目录中创建摘要文件
    create_summary_file(data, base_output_dir)
    
    print(f"\n✓ 成功转换 {success_count}/{len(data)} 个运动序列")
    print(f"输出文件保存至: {base_output_dir}/")
    
    # 从第一个运动中提取计数用于摘要
    joint_count = None
    body_count = None
    if data:
        first_motion = next(iter(data.values()))
        joint_count = first_motion['joint_pos'].shape[1]
        body_count = first_motion['body_pos_w'].shape[1]
    
    return success_count > 0, len(data), joint_count, body_count

def convert_single_motion(motion_name, motion_data, output_dir):
    """
    将单个运动序列转换为各种格式
    
    参数:
    - motion_name: 运动名称
    - motion_data: 运动数据字典
    - output_dir: 输出目录路径
    
    返回:
    - 转换是否成功
    """
    
    try:
        # 提取所有可用的数据数组
        # 关节位置: (时间步, 29) - 包含29个关节点的位置信息
        joint_pos = motion_data['joint_pos']
        # 关节速度: (时间步, 29) - 包含29个关节点的速度信息
        joint_vel = motion_data['joint_vel']
        # 世界坐标系下的身体位置: (时间步, 14, 3) - 14个身体部位的3D坐标
        body_pos_w = motion_data['body_pos_w']
        # 世界坐标系下的身体四元数: (时间步, 14, 4) - 14个身体部位的旋转四元数
        body_quat_w = motion_data['body_quat_w']
        # 世界坐标系下的身体线速度: (时间步, 14, 3) - 14个身体部位的线速度
        body_lin_vel_w = motion_data['body_lin_vel_w']
        # 世界坐标系下的身体角速度: (时间步, 14, 3) - 14个身体部位的角速度
        body_ang_vel_w = motion_data['body_ang_vel_w']
        
        # 获取时间步数
        timesteps = joint_pos.shape[0]
        print(f"  时间步数: {timesteps}, 关节数: {joint_pos.shape[1]}, 身体部件数: {body_pos_w.shape[1]}")
        
        # 1. 保存关节数据为CSV格式
        joint_pos_file = os.path.join(output_dir, "joint_pos.csv")
        save_array_as_csv(joint_pos, joint_pos_file,
                         [f"joint_{i}" for i in range(joint_pos.shape[1])])
        
        joint_vel_file = os.path.join(output_dir, "joint_vel.csv")
        save_array_as_csv(joint_vel, joint_vel_file,
                         [f"joint_vel_{i}" for i in range(joint_vel.shape[1])])
        
        # 2. 保存身体位置数据 (重塑为2D用于CSV)
        # 将形状从(timesteps, 14, 3)重塑为(timesteps, 42)，即14*3个列
        body_pos_reshaped = body_pos_w.reshape(timesteps, -1)
        body_pos_file = os.path.join(output_dir, "body_pos.csv")
        # 生成列标题，格式为: body_0_x, body_0_y, body_0_z, body_1_x...
        body_pos_headers = [f"body_{i//3}_{'xyz'[i%3]}" for i in range(body_pos_reshaped.shape[1])]
        save_array_as_csv(body_pos_reshaped, body_pos_file, body_pos_headers)
        
        # 3. 保存身体四元数数据 (重塑为2D用于CSV)
        # 将形状从(timesteps, 14, 4)重塑为(timesteps, 56)，即14*4个列
        body_quat_reshaped = body_quat_w.reshape(timesteps, -1)
        body_quat_file = os.path.join(output_dir, "body_quat.csv")
        # 生成列标题，格式为: body_0_w, body_0_x, body_0_y, body_0_z, body_1_w...
        body_quat_headers = [f"body_{i//4}_{'wxyz'[i%4]}" for i in range(body_quat_reshaped.shape[1])]
        save_array_as_csv(body_quat_reshaped, body_quat_file, body_quat_headers)
        
        # 4. 保存身体线速度数据
        # 将形状从(timesteps, 14, 3)重塑为(timesteps, 42)
        body_lin_vel_reshaped = body_lin_vel_w.reshape(timesteps, -1)
        body_lin_vel_file = os.path.join(output_dir, "body_lin_vel.csv")
        # 生成列标题，格式为: body_0_vel_x, body_0_vel_y, body_0_vel_z...
        body_lin_vel_headers = [f"body_{i//3}_vel_{'xyz'[i%3]}" for i in range(body_lin_vel_reshaped.shape[1])]
        save_array_as_csv(body_lin_vel_reshaped, body_lin_vel_file, body_lin_vel_headers)
        
        # 5. 保存身体角速度数据
        # 将形状从(timesteps, 14, 3)重塑为(timesteps, 42)
        body_ang_vel_reshaped = body_ang_vel_w.reshape(timesteps, -1)
        body_ang_vel_file = os.path.join(output_dir, "body_ang_vel.csv")
        # 生成列标题，格式为: body_0_angvel_x, body_0_angvel_y, body_0_angvel_z...
        body_ang_vel_headers = [f"body_{i//3}_angvel_{'xyz'[i%3]}" for i in range(body_ang_vel_reshaped.shape[1])]
        save_array_as_csv(body_ang_vel_reshaped, body_ang_vel_file, body_ang_vel_headers)
        
        # 6. 保存元数据
        metadata_file = os.path.join(output_dir, "metadata.txt")
        save_metadata(motion_name, motion_data, metadata_file)
        
        # 7. 保存详细信息
        info_file = os.path.join(output_dir, "info.txt")
        save_motion_info(motion_name, motion_data, info_file)
        
        print(f"  ✓ 已为 {motion_name} 保存7个文件 (关节 + 全身运动学)")
        return True
        
    except Exception as e:
        print(f"  ✗ 处理 {motion_name} 时出错: {e}")
        return False

def save_array_as_csv(array, filename, headers=None):
    """
    将numpy数组保存为CSV文件
    
    参数:
    - array: 要保存的numpy数组
    - filename: 输出文件名
    - headers: 列标题列表
    """
    import numpy as np
    
    with open(filename, 'w') as f:
        # 写入头部
        if headers:
            f.write(",".join(headers) + "\n")
        else:
            f.write(",".join([f"col_{i}" for i in range(array.shape[1])]) + "\n")
        
        # 写入数据
        for row in array:
            f.write(",".join([f"{val:.6f}" for val in row]) + "\n")

def save_metadata(motion_name, motion_data, filename):
    """
    保存元数据和索引信息
    
    参数:
    - motion_name: 运动名称
    - motion_data: 运动数据字典
    - filename: 输出文件名
    """
    
    with open(filename, 'w') as f:
        f.write(f"元数据: {motion_name}\n")
        f.write("=" * 30 + "\n\n")
        
        # 如果可用则保存身体索引
        if '_body_indexes' in motion_data:
            f.write("身体部件索引:\n")
            f.write(f"{motion_data['_body_indexes']}\n\n")
        
        # 保存总时间步数
        if 'time_step_total' in motion_data:
            f.write(f"总时间步数: {motion_data['time_step_total']}\n\n")
        
        # 数据摘要
        f.write("数据数组摘要:\n")
        for key, value in motion_data.items():
            if hasattr(value, 'shape'):
                f.write(f"  {key}: {value.shape} ({value.dtype})\n")

def save_motion_info(motion_name, motion_data, filename):
    """
    保存详细的运动信息
    
    参数:
    - motion_name: 运动名称
    - motion_data: 运动数据字典
    - filename: 输出文件名
    """
    
    with open(filename, 'w') as f:
        f.write(f"运动信息: {motion_name}\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in motion_data.items():
            f.write(f"{key}:\n")
            if hasattr(value, 'shape'):
                f.write(f"  形状: {value.shape}\n")
                f.write(f"  数据类型: {value.dtype}\n")
                if value.size > 0:
                    flat_vals = value.flatten()
                    f.write(f"  范围: [{flat_vals.min():.3f}, {flat_vals.max():.3f}]\n")
                    f.write(f"  样本: {flat_vals[:5]}\n")
            else:
                f.write(f"  值: {value}\n")
            f.write("\n")

def create_summary_file(data, output_dir):
    """
    创建包含所有运动信息的摘要文件
    
    参数:
    - data: 包含所有运动数据的字典
    - output_dir: 输出目录路径
    """
    
    summary_file = os.path.join(output_dir, "motion_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("G1运动捕捉数据摘要\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"总运动序列数: {len(data)}\n\n")
        
        # 详细的运动列表
        f.write("详细运动列表:\n")
        for motion_name, motion_data in data.items():
            joint_pos = motion_data['joint_pos']
            f.write(f"  {motion_name}:\n")
            f.write(f"    时间步数: {joint_pos.shape[0]}\n")
            f.write(f"    关节数: {joint_pos.shape[1]}\n")
            f.write(f"    身体部件数: {motion_data['body_pos_w'].shape[1]}\n")
            f.write("\n")

def main():
    """主函数 - 解析命令行参数并执行转换"""
    if len(sys.argv) < 2:
        print("使用方法: python3 convert_motions.py <pkl_file> [output_base_dir]")
        print("示例:")
        print("  python3 convert_motions.py bones_072925_test.pkl")
        print("  python3 convert_motions.py bones_072925_test.pkl custom_output/")
        print("")
        print("默认输出结构: reference/{pkl_name}/{motion_name}/")
        return
    
    pkl_file = sys.argv[1]
    output_base_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(pkl_file):
        print(f"错误: 文件不存在: {pkl_file}")
        return
    
    # 提取pkl名称用于输出消息
    pkl_name = os.path.splitext(os.path.basename(pkl_file))[0]
    
    print("G1运动数据转换器")
    print("========================")
    
    success, motion_count, joint_count, body_count = convert_motion_data(pkl_file, output_base_dir)
    if success:
        print("\n✓ 转换成功完成!")
        print("\n为每个运动提取的数据:")
        print(f"- 关节位置和速度 ({joint_count} 个关节)")
        print(f"- 世界坐标系下的身体位置 ({body_count} 个身体部件)")
        print("- 身体方向 (四元数)")
        print("- 身体线速度和角速度")
        print("- 元数据和身体部件索引")
        print("\n下一步:")
        print(f"1. 构建C++读取器: make motion_data_reader")
        print(f"2. 测试读取: ./bin/motion_data_reader reference/{pkl_name}/")
        print("3. 在您的G1控制程序中使用完整的运动学数据")
        print("\n创建的文件结构:")
        print(f"reference/{pkl_name}/")
        print("├── [motion_name_1]/")
        print("│   ├── joint_pos.csv")
        print("│   ├── joint_vel.csv")
        print("│   ├── body_pos.csv")
        print("│   ├── body_quat.csv")
        print("│   ├── body_lin_vel.csv")
        print("│   ├── body_ang_vel.csv")
        print("│   ├── metadata.txt")
        print("│   └── info.txt")
        print("├── [motion_name_2]/")
        print("├── motion_summary.txt")
        print(f"└── ... (共{motion_count}个运动文件夹)")
    else:
        print("\n✗ 转换失败")

if __name__ == "__main__":
    main()