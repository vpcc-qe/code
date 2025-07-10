#!/usr/bin/env python3
"""
Open3D颜色映射脚本
将origin.ply的颜色信息通过最近邻搜索映射到A、B、C、D四个PLY文件上
"""

import open3d as o3d
import numpy as np
import time
from typing import List, Tuple

def load_ply_with_colors(filepath: str) -> o3d.geometry.PointCloud:
    """加载带颜色的PLY文件"""
    print(f"加载文件: {filepath}")
    pcd = o3d.io.read_point_cloud(filepath)
    
    if len(pcd.colors) == 0:
        print(f"警告: {filepath} 没有颜色信息")
    else:
        print(f"成功加载 {len(pcd.points)} 个点，{len(pcd.colors)} 个颜色")
    
    return pcd

def map_colors_kdtree(source_pcd: o3d.geometry.PointCloud, target_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    使用KDTree将source_pcd的颜色映射到target_pcd
    
    Args:
        source_pcd: 有颜色信息的源点云 (origin.ply)
        target_pcd: 需要添加颜色的目标点云
    
    Returns:
        添加了颜色的目标点云
    """
    print("构建KDTree...")
    # 构建KDTree用于快速最近邻搜索
    kdtree = o3d.geometry.KDTreeFlann(source_pcd)
    
    target_points = np.asarray(target_pcd.points)
    source_colors = np.asarray(source_pcd.colors)
    
    # 为目标点云分配颜色数组
    target_colors = np.zeros((len(target_points), 3))
    
    print(f"开始颜色映射，处理 {len(target_points)} 个点...")
    
    # 对每个目标点找到源点云中最近的点
    for i, point in enumerate(target_points):
        if i % 10000 == 0:  # 每处理10000个点打印一次进度
            print(f"进度: {i}/{len(target_points)} ({i/len(target_points)*100:.1f}%)")
        
        # 搜索最近邻点 (k=1表示只找1个最近的点)
        [k, idx, _] = kdtree.search_knn_vector_3d(point, 1)
        
        if k > 0:  # 如果找到了最近邻点
            # 复制最近邻点的颜色
            target_colors[i] = source_colors[idx[0]]
    
    # 将颜色赋值给目标点云
    target_pcd.colors = o3d.utility.Vector3dVector(target_colors)
    
    print("颜色映射完成!")
    return target_pcd

def process_single_file(source_pcd: o3d.geometry.PointCloud, input_filename: str, output_filename: str):
    """处理单个文件的颜色映射"""
    print(f"\n{'='*50}")
    print(f"处理文件: {input_filename} -> {output_filename}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # 加载目标文件
    target_pcd = o3d.io.read_point_cloud(input_filename)
    print(f"目标文件包含 {len(target_pcd.points)} 个点")
    
    # 执行颜色映射
    colored_pcd = map_colors_kdtree(source_pcd, target_pcd)
    
    # 保存结果
    success = o3d.io.write_point_cloud(output_filename, colored_pcd)
    
    end_time = time.time()
    
    if success:
        print(f"成功保存到: {output_filename}")
        print(f"处理时间: {end_time - start_time:.2f} 秒")
    else:
        print(f"保存失败: {output_filename}")

def main():
    """主函数"""
    print("Open3D颜色映射脚本启动")
    print("="*60)
    
    # 文件列表
    source_file = "origin.ply"
    target_files = ["A.ply", "B.ply", "C.ply", "D.ply"]
    output_files = ["A-1.ply", "B-1.ply", "C-1.ply", "D-1.ply"]
    
    # 加载源文件 (origin.ply)
    print("加载源文件 (origin.ply)...")
    source_pcd = load_ply_with_colors(source_file)
    
    # 检查源文件是否有颜色信息
    if len(source_pcd.colors) == 0:
        print("错误: origin.ply 文件没有颜色信息!")
        return
    
    print(f"源文件包含 {len(source_pcd.points)} 个点和 {len(source_pcd.colors)} 个颜色")
    
    # 处理每个目标文件
    total_start_time = time.time()
    
    for i, (target_file, output_file) in enumerate(zip(target_files, output_files)):
        try:
            process_single_file(source_pcd, target_file, output_file)
        except Exception as e:
            print(f"处理文件 {target_file} 时出错: {str(e)}")
            continue
    
    total_end_time = time.time()
    
    print(f"\n{'='*60}")
    print("所有文件处理完成!")
    print(f"总处理时间: {total_end_time - total_start_time:.2f} 秒")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 