import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import MinkowskiEngine as ME
from MinkowskiEngine import utils as ME_utils
from data import (
    PointCloudDataset,
    position_loss,
    plot_loss_curve,
    check_convergence,
    save_point_cloud_as_ply
    
)
from network import AustinNet,SimpleAustinNet
from logger import logger
import time
import os
import open3d as o3d

torch.cuda.set_device(0)
DEVICE = 'cuda:0'
device = 'cuda'
device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'

def train_model(model, train_dataloader, optimizer, scheduler, epochs,dynamic_enabled):        
    logger.info(f"当前默认CUDA设备: {torch.cuda.current_device()}")
    model = model.to(DEVICE)
    logger.info(f"模型所在设备: {next(model.parameters()).device}")
    model.train()
    
    train_pred_loss_history = []
    train_baseline_loss_history = []
    train_pred_atob_loss_history = [] 
    train_pred_btoa_loss_history = []  
    val_pred_loss_history = []
    val_baseline_loss_history = []
    
    
    use_vpcc = False
    
    if not dynamic_enabled:
        logger.info("Dynamic Enhancement 未启用。")
    else:
        logger.info("Dynamic Enhancement启用。")
    
    
    for epoch in range(epochs):
        total_pred_loss = 0
        total_pred_atob_loss = 0
        total_pred_btoa_loss = 0
        total_baseline_loss = 0
        num_batches = 0
        logger.info(f"\nStart epoch {epoch}")
        
        if dynamic_enabled and not use_vpcc and epoch > 0 and check_convergence(train_pred_loss_history):
            use_vpcc = True
            logger.info(f"\n检测到损失收敛，从epoch {epoch} 开始启用dynamic模式")
        
        for batch_idx, batch_data in enumerate(train_dataloader):
            compress_tensor = batch_data['compress_tensor'].squeeze(0).to(device)#.reshape(-1, 3).float().to(DEVICE)
            origin_tensor = batch_data['origin_tensor'].squeeze(0).to(device)
            new_origin_tensor = batch_data['new_origin_tensor'].squeeze(0).to(device)#.reshape(-1, 4).int().to(DEVICE)      
            new_compress_tensor = batch_data['new_compress_tensor'].squeeze(0).to(device)#.reshape(-1, 4).int().to(DEVICE) 
            current_file = batch_data['filename'][0]
       
            compress_coords = ME_utils.batched_coordinates([compress_tensor], device=device)
            origin_coords = ME_utils.batched_coordinates([origin_tensor], device=device)
            new_origin_coords = ME_utils.batched_coordinates([new_origin_tensor], device=device)
            new_compress_coords = ME_utils.batched_coordinates([new_compress_tensor], device=device)
                   
           
            compress_sparse = ME.SparseTensor(
                    features=compress_tensor,
                    coordinates=compress_coords,
                    device=device
            )
            
            if compress_sparse.F.shape[0] < 10:
                print(f"警告: Batch {batch_idx} 特征数量太少，跳过")
                continue            

            pred = model(compress_sparse)
            
            pred_loss, pred_atob_loss,pred_btoa_loss,baseline_loss = position_loss(
                current_file,
                pred.F.float(), 
                compress_tensor,
                origin_tensor,
                new_origin_tensor,
                new_compress_tensor,
                use_vpcc=use_vpcc,
                current_epoch=epoch,
                cache_update_frequency=10
            )          
#             if pred_loss is None:
#                 logger.info(f'Skipping batch for {current_file} at batch_idx {batch_idx} due to high prediction loss.')
#                 continue
       
     
            # 反向传播和优化
            optimizer.zero_grad()
            pred_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            # 记录损失
            total_pred_loss += pred_loss.item()
            total_pred_atob_loss += pred_atob_loss.item()
            total_pred_btoa_loss += pred_btoa_loss.item() 
            total_baseline_loss += baseline_loss.item()
            num_batches += 1

            if batch_idx % 1 == 0:
                logger.info(f'Batch {batch_idx}, Pred Loss: {pred_loss.item():.4f}, Baseline Loss: {baseline_loss.item():.4f}')

        # 计算平均损失
        avg_train_pred_loss = total_pred_loss / num_batches
        avg_train_baseline_loss = total_baseline_loss / num_batches
        avg_train_pred_atob_loss = total_pred_atob_loss / num_batches  
        avg_train_pred_btoa_loss = total_pred_btoa_loss / num_batches 

        train_pred_loss_history.append(avg_train_pred_loss)
        train_pred_atob_loss_history.append(avg_train_pred_atob_loss)  
        train_pred_btoa_loss_history.append(avg_train_pred_btoa_loss)  
        train_baseline_loss_history.append(avg_train_baseline_loss)
        
        logger.info(f'\nEpoch {epoch + 1}/{epochs}, Pred Loss: {avg_train_pred_loss:.4f}, Baseline Loss: {avg_train_baseline_loss:.4f}')
      
    # 定期保存检查点和绘制损失曲线 - 每10个epoch
        if (epoch + 1) % 1 == 0:
            plot_loss_curve(
                        pred_losses=train_pred_loss_history,
                        baseline_losses=train_baseline_loss_history,
                        pred_atob_losses=train_pred_atob_loss_history,
                        pred_btoa_losses=train_pred_btoa_loss_history,
                        save_path='train_loss_curve.png',
                        current_epoch=epoch+1)
               


                # 保存定期检查点
            save_path = f'models/epoch_{epoch + 1}_model.pth'
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_pred_loss': avg_train_pred_loss,
                    'train_pred_atob_loss': avg_train_pred_atob_loss,  
                    'train_pred_btoa_loss': avg_train_pred_btoa_loss, 
                    'train_baseline_loss': avg_train_baseline_loss,
                    'train_pred_loss_history': train_pred_loss_history,
                    'train_pred_atob_loss_history': train_pred_atob_loss_history,  
                    'train_pred_btoa_loss_history': train_pred_btoa_loss_history, 
                    'train_baseline_loss_history': train_baseline_loss_history,
                    'val_pred_loss_history': val_pred_loss_history,
                    'val_baseline_loss_history': val_baseline_loss_history
                }, save_path)
        scheduler.step() 

    return (train_pred_loss_history, train_baseline_loss_history, 
            train_pred_atob_loss_history, train_pred_btoa_loss_history,
            val_pred_loss_history, val_baseline_loss_history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--dynamic', action='store_true')
    args = parser.parse_args()
    epochs = 100
    base_path = "/home/jupyter-haoyu/code/dataset/r1/train_dataset"

    train_dataset = PointCloudDataset(
        f"{args.dataset}/block_interpolate",
        f"{args.dataset}/block_predict_residual",
        f"{args.dataset}/block_new_predict_residual",
        f"{args.dataset}/block_new_predict_residual"
    )

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=1,
        shuffle=False
    )
    
    
    # 初始化模型和优化器
    model = SimpleAustinNet()
    model = model.float()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam([{"params": model.parameters(), 'lr': 0.001}],
                                 betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 加载上次的checkpoint
#     checkpoint = torch.load('models/epoch_13_model.pth')
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 训练模型
    train_model(model, train_dataloader, optimizer, scheduler, epochs,dynamic_enabled=args.dynamic)
    
 