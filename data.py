import os
import re
import time
import numpy as np
import pandas as pd
import torch
import open3d as o3d
import MinkowskiEngine as ME
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from scipy.spatial import KDTree
from logger import logger
import matplotlib.pyplot as plt
import time
import tempfile
import subprocess



def extract_msef_value(file_path):
    msef = None
    with open(file_path, 'r') as file:
        for line in file:
            if "mseF      (p2point)" in line:
                msef = float(line.split(':')[1].strip())
                break
    return msef

def read_psnr_from_file(filename):
    """读取单个文件中的两个 PSNR 值"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
            # 定义两个模式
            pattern1 = r'mse1,PSNR \(p2point\):\s*(\d+\.\d+)'
            pattern2 = r'mse2,PSNR \(p2point\):\s*(\d+\.\d+)'
            
            # 查找两个匹配
            match1 = re.search(pattern1, content)
            match2 = re.search(pattern2, content)
            
            # 返回一个元组，包含两个 PSNR 值
            psnr1 = float(match1.group(1)) if match1 else None
            psnr2 = float(match2.group(1)) if match2 else None
            
            return (psnr1, psnr2)
            
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return (None, None)


# a to B 和 B to A
def extract_psnr_values(file_path):
    mse1_psnr = None
    mse2_psnr = None

    with open(file_path, 'r') as file:
        for line in file:
            if "mse1,PSNR (p2point)" in line:
                mse1_psnr = float(line.split(':')[1].strip())
            elif "mse2,PSNR (p2point)" in line:
                mse2_psnr = float(line.split(':')[1].strip())

    return mse1_psnr, mse2_psnr


def extract_mse_values(file_path):
    """
    从 VPCC 输出文件中提取 mse1 和 mse2 的值
    Args:
        file_path: VPCC 输出的文本文件路径
    Returns:
        tuple: (mse1, mse2) 值，如果未找到则返回 (None, None)
    """
    mse1 = None
    mse2 = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if "mse1      (p2point):" in line:
                    mse1 = float(line.split(':')[1].strip())
                elif "mse2      (p2point):" in line:
                    mse2 = float(line.split(':')[1].strip())
        return mse1, mse2
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None
    
def plot_loss_curve(pred_losses, baseline_losses, pred_atob_losses=None, pred_btoa_losses=None, save_path='loss_curve.png', current_epoch=None):
    """
    绘制多条损失曲线
    
    Args:
        pred_losses: list, 预测重建的损失值列表
        baseline_losses: list, baseline的损失值列表
        pred_atob_losses: list, A到B预测的损失值列表
        pred_btoa_losses: list, B到A预测的损失值列表
        save_path: str, 保存图像的路径
        current_epoch: int, 当前的epoch数
    """
    plt.figure(figsize=(12, 8))
    
    # 绘制预测重建损失曲线
    plt.plot(pred_losses, 'b-', label='Prediction Loss', linewidth=2)
    
    # 绘制baseline损失曲线
    plt.plot(baseline_losses, 'r--', label='Baseline Loss', linewidth=2)
    
    # 绘制A到B预测损失曲线（如果提供）
    if pred_atob_losses:
        plt.plot(pred_atob_losses, 'g-', label='A to B Loss', linewidth=2)
    
    # 绘制B到A预测损失曲线（如果提供）
    if pred_btoa_losses:
        plt.plot(pred_btoa_losses, 'm-', label='B to A Loss', linewidth=2)
    
    # 设置图表属性
    plt.title(f'Training Losses (Current Epoch: {current_epoch})' if current_epoch is not None else 'Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 设置y轴范围
    all_losses = [loss for loss_list in [pred_losses, baseline_losses, pred_atob_losses or [], pred_btoa_losses or []] for loss in loss_list if loss_list]
    if all_losses:
        min_loss = min(all_losses)
        max_loss = max(all_losses)
        plt.ylim([max(0, min_loss * 0.9), max_loss * 1.1])
    
    plt.savefig(save_path)
    plt.close()


class MyKDTree:
    def __init__(self, data):
        self.data = np.array(data)
        self.n_points, self.dim = self.data.shape
        self.tree = self._build_tree(np.arange(self.n_points))

    class Node:
        def __init__(self, index, split_dim, left, right):
            self.index = index
            self.split_dim = split_dim
            self.left = left
            self.right = right

    def _build_tree(self, indices):
        if len(indices) == 0:
            return None
        if len(indices) == 1:
            return self.Node(indices[0], None, None, None)

        # 选择切分维度
        split_dim = np.argmax(self.data[indices].ptp(axis=0))  # 最大范围维度
        indices = indices[np.argsort(self.data[indices, split_dim])]  # 排序

        median_idx = len(indices) // 2
        return self.Node(
            index=indices[median_idx],
            split_dim=split_dim,
            left=self._build_tree(indices[:median_idx]),
            right=self._build_tree(indices[median_idx + 1 :]),
        )

    def _nearest(self, point, node, best_dist, best_index):
        if node is None:
            return best_dist, best_index

        # 计算当前点到目标点的距离
        dist = np.linalg.norm(point - self.data[node.index])
        if dist < best_dist:
            best_dist = dist
            best_index = node.index

        # 如果是叶子节点，直接返回
        if node.split_dim is None:
            return best_dist, best_index

        # 确定搜索顺序
        diff = point[node.split_dim] - self.data[node.index, node.split_dim]  # 确保 diff 是标量
        first, second = (node.left, node.right) if diff < 0 else (node.right, node.left)

        # 搜索较近的子树
        best_dist, best_index = self._nearest(point, first, best_dist, best_index)

        # 检查是否需要搜索另一个子树
        if abs(diff) < best_dist:
            best_dist, best_index = self._nearest(point, second, best_dist, best_index)

        return best_dist, best_index

    def query(self, point):
        return self._nearest(point, self.tree, float("inf"), -1)

    
def find_nearest_neighbors(pointcloudA, pointcloudB):
    # 记住原始设备
    device = pointcloudA.device if torch.is_tensor(pointcloudA) else None
    
    # 首先将张量移动到 CPU 并分离计算图
    if torch.is_tensor(pointcloudA):
        pointcloudA = pointcloudA.detach().cpu()
    if torch.is_tensor(pointcloudB):
        pointcloudB = pointcloudB.detach().cpu()
    
    # 转换为 NumPy 数组
    pointcloudA = pointcloudA.numpy() if torch.is_tensor(pointcloudA) else pointcloudA
    pointcloudB = pointcloudB.numpy() if torch.is_tensor(pointcloudB) else pointcloudB
    
    # 构建 KD 树并查找最近邻
    kd_tree = KDTree(pointcloudB)
    nearest_points = np.array([kd_tree.data[kd_tree.query(p)[1]] for p in pointcloudA])
    
    # 将结果转换回 PyTorch tensor 并移回原始设备
    nearest_points = torch.from_numpy(nearest_points)
    if device is not None:
        nearest_points = nearest_points.to(device)
    
    return nearest_points



def position_total_loss(pred, origin, compress, batch_data):
    """
    计算整个文件的position loss
    
    Args:
        pred: 当前batch的预测坐标 [N, 3]
        origin: 当前batch的原始坐标 [N, 3]
        compress: 当前batch的压缩坐标 [N, 3]
        batch_data: 包含文件索引等信息的字典
    """
    total_pred_loss = 0
    total_baseline_loss = 0
    batch_size = len(batch_data['file_index'])
    
    # 对batch中的每个样本分别处理
    for i in range(batch_size):
        current_file_index = batch_data['file_index'][i]
        current_block_index = batch_data['block_index'][i]
        
        # 找出同一个文件的所有block
        same_file_mask = (batch_data['file_index'] == current_file_index)
        
        # 获取当前block的预测结果
        current_pred = pred[i].unsqueeze(0)  # [1, 3]
        
        # 构建完整的点云
        # 1. 对于预测点云，使用当前block的预测结果，其他block使用compress
        full_pred = []
        for j in range(batch_size):
            if batch_data['file_index'][j] == current_file_index:
                if batch_data['block_index'][j] == current_block_index:
                    full_pred.append(current_pred)
                else:
                    full_pred.append(compress[j].unsqueeze(0))
        
        full_pred = torch.cat(full_pred, dim=0)
        
        # 2. 获取对应的原始点云
        full_origin = origin[same_file_mask]
        
        # 3. 获取对应的压缩点云
        full_compress = compress[same_file_mask]
        
        # 计算整个文件的loss
        pred_loss = torch.nn.functional.mse_loss(full_pred, full_origin)
        baseline_loss = torch.nn.functional.mse_loss(full_compress, full_origin)
        
        total_pred_loss += pred_loss
        total_baseline_loss += baseline_loss
    
    # 返回平均loss
    return total_pred_loss / batch_size, total_baseline_loss / batch_size

def extract_vpcc_origin(source_points, target_points, resolution=1023):
    """
    使用 VPCC 的最近邻搜索逻辑来找到对应的原始点
    返回源点云(A)和目标点云(B)中的对应点
    """
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as source_file, \
         tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as target_file:
        
        source_path = source_file.name
        target_path = target_file.name
        
        source_np = source_points.detach().cpu().numpy()
        target_np = target_points.detach().cpu().numpy()
        
        save_point_cloud_as_ply(source_np, source_path)
        save_point_cloud_as_ply(target_np, target_path)
        
        cmd = f"../../mpeg-pcc-dmetric-0.13.05/test/pc_error_d --fileA={source_path} --fileB={target_path} --color=1 --resolution={resolution} --dropdups=0 --singlePass=1"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        # 使用字典存储点对应关系，键为索引
        correspondences = {}
        
        # 更新正则表达式模式，捕获索引
        pattern = r"Point A\[(\d+)\] \((-?\d+\.?\d*),(-?\d+\.?\d*),(-?\d+\.?\d*)\) -> B\[\d+\] \((-?\d+\.?\d*),(-?\d+\.?\d*),(-?\d+\.?\d*)\)"
        
        while True:
            output = process.stdout.readline()
#             print(output.strip())
            if output == '' and process.poll() is not None:
                break
            match = re.search(pattern, output)
            if match:
                # 获取索引和坐标
                index = int(match.group(1))
                source_x, source_y, source_z = map(float, match.groups()[1:4])
                target_x, target_y, target_z = map(float, match.groups()[4:])
                
#                 print(f"索引 {index}:")
#                 print("找到的坐标", source_x, source_y, source_z)
#                 print("匹配到的坐标", target_x, target_y, target_z)
#                 time.sleep(100000)
                
                # 存储对应关系
                correspondences[index] = [target_x, target_y, target_z]
        
        process.communicate()
        
        os.unlink(source_path)
        os.unlink(target_path)
        
        if not correspondences:
            raise RuntimeError("未找到对应点")
        
        # 按索引顺序构建目标点列表
        target_points_list = []
        for i in range(len(correspondences)):
            if i not in correspondences:
                raise RuntimeError(f"缺少索引 {i} 的对应点")
            target_points_list.append(correspondences[i])
        
        # 转换为tensor
        target_points_tensor = torch.tensor(target_points_list,
                                          dtype=source_points.dtype,
                                          device=source_points.device,
                                          requires_grad=True)
        
        return target_points_tensor


    

def check_convergence(loss_history, window_size=5, threshold=0.001):
    """
    检查最近几轮的损失是否收敛
    Args:
        loss_history: 损失历史记录
        window_size: 检查最近几轮
        threshold: 收敛阈值
    Returns:
        bool: 是否收敛
    """
    if len(loss_history) < window_size:
        return False
    
    recent_losses = loss_history[-window_size:]
    max_diff = max(recent_losses) - min(recent_losses)
    
    # 如果最大差值相对于平均损失小于阈值，认为已收敛
    return max_diff < threshold

import os
import torch

# 在文件开头定义全局变量
_last_cache_update_epoch = -1

class FileCache:
    def __init__(self, cache_dir="./vpcc_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_path(self, filename):
        safe_name = filename.replace('/', '_').replace('\\', '_')
        return os.path.join(self.cache_dir, f"{safe_name}.pt")
    
    def exists(self, filename):
        return os.path.exists(self.get_path(filename))
    
    def save(self, filename, tensor1, tensor2):
        path = self.get_path(filename)
        torch.save((tensor1.cpu(), tensor2.cpu()), path)
        
    def load(self, filename):
        path = self.get_path(filename)
        if os.path.exists(path):
            tensor_tuple = torch.load(path)
            return tensor_tuple[0].cuda(), tensor_tuple[1].cuda()
        return None

# 初始化文件缓存
file_cache = FileCache()


def reorder_predict(compress, new_compress, predict):
    """
    根据compress到new_compress的排序映射，重排predict
    
    参数:
    compress: 原始压缩数据
    new_compress: 重排序后的压缩数据
    predict: 需要重排序的预测数据
    
    返回:
    new_predict: 重排序后的预测数据
    """
    # 将compress和new_compress转换为可哈希的形式（如果它们是张量）
    if torch.is_tensor(compress):
        compress_tuples = [tuple(p.tolist()) for p in compress]
    else:
        compress_tuples = [tuple(p) for p in compress]
    
    if torch.is_tensor(new_compress):
        new_compress_tuples = [tuple(p.tolist()) for p in new_compress]
    else:
        new_compress_tuples = [tuple(p) for p in new_compress]
    
    # 创建从compress点到索引的映射
    compress_to_idx = {point: idx for idx, point in enumerate(compress_tuples)}
    
    # 对于new_compress中的每个点，找到它在compress中的索引
    new_indices = [compress_to_idx[point] for point in new_compress_tuples]
    
    # 使用这些索引重排predict
    if torch.is_tensor(predict):
        new_predict = predict[new_indices]
    else:
        new_predict = [predict[idx] for idx in new_indices]
    
    return new_predict


# def position_loss(current_file, pred, pred_origin, compress, origin, new_origin,new_compress, use_vpcc=False, current_epoch=None, cache_update_frequency=10):
    
#     # 声明使用全局变量
#     global _last_cache_update_epoch
#     # Step 2
#     if use_vpcc:
#         should_update_cache = (
#             current_epoch is not None and 
#             (current_epoch - _last_cache_update_epoch >= cache_update_frequency)
#         )
        
#         if should_update_cache:
#             new_origin_1 = extract_vpcc_origin(pred, origin)
#             pred_btoa_loss = torch.nn.functional.mse_loss(pred, new_origin_1)
#             new_compress_1 = extract_vpcc_origin(origin, pred)
#             pred_atob_loss = torch.nn.functional.mse_loss(origin, new_compress_1)
#             pred_loss = 0.5 * pred_btoa_loss + 0.5 * pred_atob_loss
#             file_cache.save(current_file, new_origin_1, new_compress_1)
#             logger.info(f"更新缓存：{current_file}")
#             _last_cache_update_epoch = current_epoch
            
#         elif file_cache.exists(current_file):
#             new_origin_1,new_compress_1 = file_cache.load(current_file)
#             pred_btoa_loss = torch.nn.functional.mse_loss(pred, new_origin_1)
#             pred_atob_loss = torch.nn.functional.mse_loss(origin, new_compress_1)
#             pred_loss = 0.5 * pred_btoa_loss + 0.5 * pred_atob_loss
#         else:    
#             new_origin_1 = extract_vpcc_origin(pred, origin)
#             pred_btoa_loss = torch.nn.functional.mse_loss(pred, new_origin_1)
#             new_compress_1 = extract_vpcc_origin(origin, pred)
#             pred_atob_loss = torch.nn.functional.mse_loss(origin, new_compress_1)
#             pred_loss = 0.5 * pred_btoa_loss + 0.5 * pred_atob_loss
#             file_cache.save(current_file, new_origin_1, new_compress_1)
#             logger.info(f"首次缓存：{current_file}")
#             _last_cache_update_epoch = current_epoch
#     else: 
#         # Step 1:
#         new_predict = reorder_predict(compress, new_compress, pred)
#         pred_atob_loss = torch.nn.functional.mse_loss(pred, new_origin)
#         pred_btoa_loss = torch.nn.functional.mse_loss(origin, new_predict)
#         pred_loss = 0.5 * pred_atob_loss + 0.5 * pred_btoa_loss
      
#     baseline_loss = torch.nn.functional.mse_loss(compress, new_origin)
#     return pred_loss, pred_atob_loss, pred_btoa_loss, baseline_loss

# 最原始的 position_loss only btoa
def position_loss(current_file, pred, compress, origin, new_origin,new_origin_atob, use_vpcc=False, current_epoch=None, cache_update_frequency=10):
    
    # 声明使用全局变量
    global _last_cache_update_epoch
    # Step 2:
    if False:
        should_update_cache = (
            current_epoch is not None and 
            (current_epoch - _last_cache_update_epoch >= cache_update_frequency)
        )
        
        if should_update_cache:
            pred_origin = extract_vpcc_origin(pred, origin)
            pred_loss = torch.nn.functional.mse_loss(pred, pred_origin)
            file_cache.save(current_file, pred_origin, pred_origin)
            logger.info(f"更新缓存：{current_file}")
            _last_cache_update_epoch = current_epoch
            
        elif file_cache.exists(current_file):
            pred_origin,pred_origin_atob = file_cache.load(current_file)
            pred_loss = torch.nn.functional.mse_loss(pred, pred_origin)
        else:    
            pred_origin = extract_vpcc_origin(pred, origin)
            pred_loss = torch.nn.functional.mse_loss(pred, pred_origin)
            file_cache.save(current_file, pred_origin,pred_origin)
            logger.info(f"首次缓存：{current_file}")
            _last_cache_update_epoch = current_epoch
    else:    
        pred_loss = torch.nn.functional.mse_loss(pred, new_origin)
            
    baseline_loss = torch.nn.functional.mse_loss(compress, new_origin)
    
    return pred_loss, pred_loss,pred_loss,baseline_loss


def check_duplicates(points, name="points"):
    """
    使用 open3d 检查点云中的重复点
    Args:
        points: tensor [N, 3] 或 [N, 4]
        name: 点云名称，用于打印信息
    Returns:
        tuple: (has_duplicates, num_unique, num_total, duplicate_points)
    """
    
    # 确保是 numpy 数组并且只取 xyz 坐标
    if isinstance(points, torch.Tensor):
        points_np = points.detach().cpu().numpy()
    else:
        points_np = points
        
    # 只取前三列（xyz坐标）
    if points_np.shape[1] > 3:
        points_np = points_np[:, :3]
    
    # 创建 open3d 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    
    # 获取原始点数
    num_total = len(points_np)
    
    # 使用 open3d 的去重功能
    pcd_no_duplicates = pcd.voxel_down_sample(voxel_size=0.0000001)  # 使用极小的体素尺寸来确保只去除完全重叠的点
    num_unique = len(pcd_no_duplicates.points)
    
    has_duplicates = num_unique < num_total
    
    print(f"\nChecking {name} point cloud:")
    print(f"Total points: {num_total}")
    print(f"Unique points: {num_unique}")
    print(f"Duplicate points: {num_total - num_unique}")
    print(f"Has duplicates: {has_duplicates}")
    
    # 如果有重复点，找出具体的重复点
    duplicate_points = []
    if has_duplicates:
        # 将点转换为元组以便比较
        points_set = set()
        points_count = {}
        
        for point in points_np:
            point_tuple = tuple(point)
            if point_tuple in points_set:
                points_count[point_tuple] = points_count.get(point_tuple, 1) + 1
                if points_count[point_tuple] == 2:  # 第一次发现重复时添加
                    duplicate_points.append(point)
            else:
                points_set.add(point_tuple)
        
        # 打印一些重复点的例子
        print("\nExample duplicate points (first 5):")
        for i, point in enumerate(duplicate_points[:5]):
            print(f"Point {i+1}: ({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})")
            print(f"Appears {points_count[tuple(point)]} times")
    
    return has_duplicates, num_unique, num_total, duplicate_points


def format_vpcc_output(output_bytes):
    """
    格式化VPCC输出，移除转义字符并整理格式
    """
    # 将bytes转换为字符串
    output_str = output_bytes.decode()
    
    # 按行分割并移除空行
    lines = [line.strip() for line in output_str.split('\n') if line.strip()]
    
    # 整理输出
    formatted_output = "=== VPCC Metrics Report ===\n"
    
    for line in lines:
        # 移除多余的空格和制表符
        line = ' '.join(line.split())
        
        # 根据内容添加适当的缩进和分类
        if "Parameters" in line:
            formatted_output += "\n" + line + "\n"
        elif line.startswith('+'):
            continue
        elif line.startswith('computeChecksum') or line.startswith('compute') or line.startswith('start') or line.startswith('frame') or line.startswith('nb') or line.startswith('resolution') or line.startswith('drop'):
            formatted_output += "    " + line + "\n"
        elif "Metrics results" in line:
            formatted_output += "\n" + line + "\n"
        elif "WARNING" in line or "Point cloud sizes" in line:
            formatted_output += "\n" + line + "\n"
        elif "mse" in line or "PSNR" in line:
            formatted_output += "    " + line + "\n"
        else:
            formatted_output += line + "\n"
    
    return formatted_output


def position_psnr_loss(pred, origin, compress):
    """
    保持梯度的双向位置损失计算
    """
    def compute_unidirectional_mse(source, target):
        """可微分的单向 MSE 计算"""
        # 构建成对距离矩阵
        diff = source.unsqueeze(1) - target.unsqueeze(0)  # [N, M, 3]
        dist = torch.sum(diff ** 2, dim=2)  # [N, M]
        
        # 对每个源点找最近的目标点
        min_dist, _ = torch.min(dist, dim=1)  # [N]
        
        return torch.mean(min_dist)

    def compute_bidirectional_mse(cloud1, cloud2):
        """计算双向 MSE"""
        # 确保输入是浮点类型
        cloud1 = cloud1.float()
        cloud2 = cloud2.float()
        
        # 计算双向距离
        mse_1to2 = compute_unidirectional_mse(cloud1, cloud2)
        mse_2to1 = compute_unidirectional_mse(cloud2, cloud1)
        
        # 取最大值
        return torch.max(mse_1to2, mse_2to1)

    # 计算损失
    pred_loss = compute_bidirectional_mse(pred, origin)
    baseline_loss = compute_bidirectional_mse(compress, origin)
    
    return pred_loss, baseline_loss




def remove_duplicates(points):
    """
    去除点云数据中的重复点。

    参数:
    points (numpy.ndarray): 点云数据，形状为 (N, D)。

    返回:
    numpy.ndarray: 去除重复点后的点云数据。
    """
    df = pd.DataFrame(points)
    df = df.drop_duplicates()
    return df.to_numpy()

def has_duplicates(points, tol=1e-9):
    """
    检查点云数据集中是否存在重复的点。

    参数:
    points (torch.Tensor or numpy.ndarray): 点云数据，形状为 (N, D)。
    tol (float): 判断重复点的容差，默认为 1e-9。

    返回:
    bool: 如果存在重复的点，则返回 True；否则返回 False。
    """
    if isinstance(points, torch.Tensor):
        # 如果是 GPU 张量，先移动到 CPU
        if points.is_cuda:
            points = points.cpu()
        # 转换为 NumPy 数组
        points = points.numpy()

    tree = KDTree(points)
    for i, point in enumerate(points):
        distances, indices = tree.query(point, k=2)
        if distances[1] < tol:
            return True
    return False


def has_duplicates_output(points, tol=1e-9):
    """
    检查点云数据集中是否存在重复的点，并输出重复的坐标。

    参数:
    points (torch.Tensor or numpy.ndarray): 点云数据，形状为 (N, D)。
    tol (float): 判断重复点的容差，默认为 1e-9。

    返回:
    tuple: (bool, list)，如果存在重复的点，则返回 (True, 重复点的列表)；否则返回 (False, 空列表)。
    """
    if isinstance(points, torch.Tensor):
        # 如果是 GPU 张量，先移动到 CPU
        if points.is_cuda:
            points = points.cpu()
        # 转换为 NumPy 数组
        points = points.numpy()

    tree = KDTree(points)
    duplicates = []
    for i, point in enumerate(points):
        distances, indices = tree.query(point, k=2)
        if distances[1] < tol:
            duplicates.append(point)

    has_dup = len(duplicates) > 0
    return has_dup, duplicates
    
def find_corresponding_original_points(compressed_points, original_points):
    """
    為壓縮點雲中的每個點找到原始點雲中未被使用的最近點
    
    參數:
    compressed_points: 壓縮後的點雲數據 (N, D)
    original_points: 原始點雲數據 (M, D)
    
    返回:
    numpy.ndarray: 與壓縮點雲相同shape的矩陣，包含來自原始點雲的未重複點
    
    異常:
    ValueError: 當無法找到足夠的唯一對應點時拋出
    """
    # 首先驗證基本條件
    if len(compressed_points) > len(original_points):
        raise ValueError(
            f"壓縮點數量({len(compressed_points)})不能大於原始點數量({len(original_points)})"
        )
    
    # 將原始點轉換為tuple以便使用set操作
    original_points_set = set(map(tuple, original_points))
    if len(original_points_set) < len(compressed_points):
        raise ValueError(
            f"原始點雲中的唯一點數量({len(original_points_set)})小於壓縮點數量({len(compressed_points)})"
        )
    
    tree = KDTree(original_points)
    result = np.zeros_like(compressed_points)
    used_original_indices = set()
    
    # 為每個壓縮點找對應的原始點
    for i, comp_point in enumerate(compressed_points):
        # 初始搜索範圍設為所有剩餘的原始點
        remaining_points = len(original_points) - len(used_original_indices)
        if remaining_points < (len(compressed_points) - i):
            raise ValueError(
                f"剩餘可用點數({remaining_points})小於待處理的壓縮點數({len(compressed_points) - i})"
            )
        
        # 查詢所有剩餘的點
        distances, indices = tree.query(comp_point, k=len(original_points))
        
        # 找到第一個未使用的點
        found = False
        for idx in indices:
            if idx not in used_original_indices:
                # 驗證這個點確實來自原始點雲
                point_tuple = tuple(original_points[idx])
                if point_tuple in original_points_set:
                    result[i] = original_points[idx]
                    used_original_indices.add(idx)
                    found = True
                    break
        
        if not found:
            raise ValueError(f"無法為壓縮點 {i} 找到未使用的對應點")
        
        # 打印進度
        if (i + 1) % 1000 == 0:
            logger.info(f"已處理: {i + 1}/{len(compressed_points)} 點")
    
    # 最終驗證
    result_points_set = set(map(tuple, result))
    
    # 驗證結果長度
    if len(result) != len(compressed_points):
        raise ValueError(
            f"結果點數({len(result)})與壓縮點數({len(compressed_points)})不匹配"
        )
    
    # 驗證沒有重複點
    if len(result_points_set) != len(result):
        raise ValueError(
            f"結果中存在重複點: 唯一點數({len(result_points_set)}) != 總點數({len(result)})"
        )
    
    # 驗證所有點都來自原始點雲
    if not result_points_set.issubset(original_points_set):
        raise ValueError("結果中包含不在原始點雲中的點")
    
    logger.info("驗證通過:")
    logger.info(f"- 結果點數: {len(result)}")
    logger.info(f"- 唯一點數: {len(result_points_set)}")
    logger.info(f"- 所有點都來自原始點雲")
    
    return result


def save_point_cloud_as_ply(coords, filename):
    """
    将点云数据保存为 PLY 文件，只保存坐标信息。
    
    Args:
        coords: (N, 3) 点云坐标
        filename: 要保存的 PLY 文件名
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)

    # 保存为 PLY 文件
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)


def match_points_o3d(points_A, points_B):
    # 将numpy数组转换为open3d的点云格式
    pcd_A = o3d.geometry.PointCloud()
    pcd_A.points = o3d.utility.Vector3dVector(points_A)
    
    # 构建KDTree
    kdtree = o3d.geometry.KDTreeFlann(pcd_A)
    
    # 初始化结果点云C，大小与B相同
    points_C = np.zeros_like(points_B)
    matched_points = set()
    
    # 对B中的每个点找A中最近的点
    for i, point in enumerate(points_B):
        k = 1  # 先找1个最近邻
        while True:
            # search_knn 返回 (查找成功与否, 最近邻索引列表, 距离列表)
            _, idx, _ = kdtree.search_knn_vector_3d(point, k)
            
            # 找到未被匹配的点
            for j in idx:
                if j not in matched_points:
                    matched_points.add(j)
                    points_C[i] = points_A[j]  # 直接存储到结果点云C中
                    break
            else:
                # 如果都已匹配，增加搜索范围
                k *= 2
                continue
            break
    
    return points_C


class PointCloudDataset(Dataset):
    def __init__(self, compress_dir, origin_dir, new_origin_dir, new_compress_dir,target_index=None):
        self.compress_dir = compress_dir
        self.origin_dir = origin_dir
        self.new_origin_dir = new_origin_dir
        self.new_compress_dir = new_compress_dir
        self.file_pairs = []
        
        # 获取所有文件
        compress_files = sorted([f for f in os.listdir(compress_dir) if f.endswith('.ply')])
        origin_files = sorted([f for f in os.listdir(origin_dir) if f.endswith('.ply')])
        new_origin_files = sorted([f for f in os.listdir(new_origin_dir) if f.endswith('.ply')])
        new_compress_files = sorted([f for f in os.listdir(new_compress_dir) if f.endswith('.ply')])
        
        def get_file_info(filename):
            match = re.search(r'_(\d+)_block_(\d+)\.ply$', filename)
            if match:
                vox_num = match.group(1)  # 例如: 0536
                block_num = int(match.group(2))  # 例如: 0
                return vox_num, block_num
            return None, None

        # 创建原始文件的查找字典
        origin_dict = {}
        new_origin_dict = {}
        new_compress_dict = {}
        for origin_file in origin_files:
            feature = get_file_info(origin_file)
            if feature:
                origin_dict[feature] = origin_file
                
        for new_origin_file in new_origin_files:
            feature = get_file_info(new_origin_file)
            if feature:
                new_origin_dict[feature] = new_origin_file
               
        for new_compress_file in new_compress_files:
            feature = get_file_info(new_compress_file)
            if feature:
                new_compress_dict[feature] = new_compress_file
                
        # 只处理目标索引的文件
        for compress_file in compress_files:
            # 如果指定了target_index，只处理包含该索引的文件
            if target_index and target_index not in compress_file:
                continue
                
            feature = get_file_info(compress_file)
            if feature and feature in origin_dict and feature in new_origin_dict:
                self.file_pairs.append((
                    compress_file,
                    origin_dict[feature],
                    new_origin_dict[feature],
                    new_compress_dict[feature]
                ))
                print(f"\n成功匹配: {compress_file} <-> {origin_dict[feature]} <-> {new_origin_dict[feature]}")
            else:
                print(f"\n警告: 未找到文件 {compress_file} 的完整匹配")

        print(f"\n总共找到 {len(self.file_pairs)} 组完整匹配的文件")
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        compress_file, origin_file, new_origin_file, new_compress_file = self.file_pairs[idx]
        print(f"加载第 {idx} 个文件: {compress_file}")
        # 加载压缩点云
        compress_pcd = o3d.io.read_point_cloud(os.path.join(self.compress_dir, compress_file))
        compress_points = np.asarray(compress_pcd.points)
        compress_tensor = torch.from_numpy(compress_points).float()
        
        # 加载原始点云
        origin_pcd = o3d.io.read_point_cloud(os.path.join(self.origin_dir, origin_file))
        origin_points = np.asarray(origin_pcd.points)
        origin_tensor = torch.from_numpy(origin_points).float()
        
        # 加载新的原始点云
        new_origin_pcd = o3d.io.read_point_cloud(os.path.join(self.new_origin_dir, new_origin_file))
        new_origin_points = np.asarray(new_origin_pcd.points)
        new_origin_tensor = torch.from_numpy(new_origin_points).float()
        
        # 加载新的原始点云
        new_compress_pcd = o3d.io.read_point_cloud(os.path.join(self.new_compress_dir, new_compress_file))
        new_compress_points = np.asarray(new_compress_pcd.points)
        new_compress_tensor = torch.from_numpy(new_compress_points).float()
        return {
            'compress_tensor': compress_tensor,
            'origin_tensor': origin_tensor,
            'new_origin_tensor': new_origin_tensor,
            'new_compress_tensor': new_compress_tensor,
            'filename': compress_file
        }
