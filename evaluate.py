import os
import subprocess
import torch
import numpy as np
import MinkowskiEngine as ME
from MinkowskiEngine import utils as ME_utils
from network import SimpleAustinNet, AustinNet, SimpleAustinNetConv2, SimpleAustinNetConv1
from logger import logger
from data import save_point_cloud_as_ply, read_psnr_from_file, PointCloudDataset
import pandas as pd

device = 'cuda'
device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'

def predict(model1_path, model2_path, compress_dir, interpolate_dir, origin_dir, output_dir, target_index, dataset_name):
    """
    使用两个训练好的模型分别对 compress_dir 和 interpolate_dir 进行预测，合并预测点云并返回 PSNR 值
    """
    # 从模型路径中提取 epoch 号
    epoch_num = os.path.basename(model1_path).split('_')[1]

    logger.info(f"开始预测 epoch {epoch_num} for {dataset_name}-{target_index}...")

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # === 1. 用 model1 预测 COMPRESS_DIR ===
    model1 = SimpleAustinNet()
    checkpoint1 = torch.load(model1_path)
    model1.load_state_dict(checkpoint1['model_state_dict'])
    model1 = model1.to(device)
    model1.eval()

    test_dataset1 = PointCloudDataset(
        compress_dir,
        compress_dir,
        compress_dir,
        compress_dir,
        target_index=target_index
    )
    dataloader1 = torch.utils.data.DataLoader(test_dataset1, batch_size=1, shuffle=False)

    all_predicted_coords1 = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader1):            
            if target_index not in batch_data['filename'][0]:
                continue
            compress_tensor = batch_data['compress_tensor'].squeeze(0).to(device)
            compress_coords = ME_utils.batched_coordinates([compress_tensor], device=device)
            compress_sparse = ME.SparseTensor(
                features=compress_tensor,
                coordinates=compress_coords,
                device=device
            )
            predict = model1(compress_sparse)
            predicted_coords = predict.F.cpu().numpy()
            all_predicted_coords1.append(predicted_coords)
            logger.info(f"完成块 {batch_data['filename']} 的 model1 预测")

    # === 2. 用 model2 预测 INTERPOLATE_DIR ===
    model2 = SimpleAustinNet()
    checkpoint2 = torch.load(model2_path)
    model2.load_state_dict(checkpoint2['model_state_dict'])
    model2 = model2.to(device)
    model2.eval()

    test_dataset2 = PointCloudDataset(
        interpolate_dir,
        interpolate_dir,
        interpolate_dir,
        interpolate_dir,
        target_index=target_index
    )
    
    dataloader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=1, shuffle=False)

    all_predicted_coords2 = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader2):            
            if target_index not in batch_data['filename'][0]:
                continue
            interpolate_tensor = batch_data['compress_tensor'].squeeze(0).to(device)
            interpolate_coords = ME_utils.batched_coordinates([interpolate_tensor], device=device)
            interpolate_sparse = ME.SparseTensor(
                features=interpolate_tensor,
                coordinates=interpolate_coords,
                device=device
            )
            predict = model2(interpolate_sparse)
            predicted_coords = predict.F.cpu().numpy()
            all_predicted_coords2.append(predicted_coords)
            logger.info(f"完成块 {batch_data['filename']} 的 model2 预测")

#     === 3. 合并两个模型的预测点云 ===
    merged_predicted_coords = np.concatenate(
        [np.concatenate(all_predicted_coords1, axis=0) if all_predicted_coords1 else np.zeros((0,3)),
         np.concatenate(all_predicted_coords2, axis=0) if all_predicted_coords2 else np.zeros((0,3))],
        axis=0
    )
    # Only Model 2
    merged_predicted_coords = np.concatenate(
        [np.concatenate(all_predicted_coords1, axis=0) if all_predicted_coords2 else np.zeros((0,3))],
        axis=0
    )
    logger.info(f"合并后的预测点云形状: {merged_predicted_coords.shape}")

    # === 4. 保存完整的预测点云 ===
    predict_file = f"{output_dir}/predict_{dataset_name}_{target_index}_{epoch_num}.ply"
    save_point_cloud_as_ply(merged_predicted_coords, predict_file)
    logger.info(f"预测结果已保存到: {predict_file}")

    # === 5. 后续流程不变 ===
    if dataset_name == "redandblack":
        origin_file = f"../../Data/8i_test/orig/redandblack/Ply/redandblack_vox10_{target_index}.ply"
        compress_file = f"../../Data/8i_test/8x/redandblack/r2/S26C03R03_rec_{target_index}.ply"
    elif dataset_name == "soldier":
        origin_file = f"../../Data/8i_test/orig/soldier/Ply/soldier_vox10_{target_index}.ply"
        compress_file = f"../../Data/8i_test/8x/soldier/r2/S26C03R03_rec_{target_index}.ply"
    else:
        raise ValueError("Unknown dataset_name")

    psnr_AtoB, psnr_BtoA = calculate_and_save_psnr(
        origin_file, compress_file, predict_file, output_dir, epoch_num, logger, dataset_name, target_index
    )

    return psnr_AtoB, psnr_BtoA

def calculate_baseline_psnr(output_dir, target_index, dataset_name):
    logger.info(f"计算 baseline PSNR for {dataset_name}-{target_index}...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if dataset_name == "redandblack":
        origin_file = f"../../Data/8i_test/orig/redandblack/Ply/redandblack_vox10_{target_index}.ply"
        compress_file = f"../../Data/8i_test/8x/redandblack/Ply/S26C03R03_rec_{target_index}.ply"
    elif dataset_name == "soldier":
        origin_file = f"../../Data/8i_test/orig/soldier/Ply/soldier_vox10_{target_index}.ply"
        compress_file = f"../../Data/8i_test/8x/soldier/Ply/S26C03R03_rec_{target_index}.ply"
    else:
        raise ValueError("Unknown dataset_name")

    try:
        cmd = f"../../mpeg-pcc-dmetric-0.13.05-origin/mpeg-pcc-dmetric-0.13.05/test/pc_error_d --fileA={origin_file} --fileB={compress_file} --resolution=1023 --dropdups=0"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, _ = process.communicate()

        psnr_file = f"{output_dir}/psnr_baseline_{dataset_name}_{target_index}.txt"
        with open(psnr_file, 'w') as f:
            f.write(output.decode())
        logger.info(f"Baseline PSNR 已保存到：{psnr_file}")

        psnr_AtoB, psnr_BtoA = read_psnr_from_file(psnr_file)
        logger.info(f"{dataset_name}-{target_index} baseline psnr A to B: {psnr_AtoB}, B to A: {psnr_BtoA}")

        return psnr_AtoB, psnr_BtoA

    except Exception as e:
        logger.error(f"Error in baseline PSNR calculation for {dataset_name}-{target_index}: {str(e)}")
        return None, None

def calculate_and_save_psnr(original_file, compressed_file, result_file, output_dir, epoch_num, logger, dataset_name, target_index):
    try:
        cmd = f"../../mpeg-pcc-dmetric-0.13.05-origin/mpeg-pcc-dmetric-0.13.05/test/pc_error_d --fileA={original_file} --fileB={result_file} --resolution=1023 --dropdups=0"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, _ = process.communicate()

        psnr_file = f"{output_dir}/psnr_predict_{dataset_name}_{target_index}_{epoch_num}.txt"
        with open(psnr_file, 'w') as f:
            f.write(output.decode())
        logger.info(f"处理结果 PSNR 已保存到：{psnr_file}")

        psnr_AtoB, psnr_BtoA = read_psnr_from_file(psnr_file)
        logger.info(f"{dataset_name}-{target_index} predict psnr A to B: {psnr_AtoB}, B to A: {psnr_BtoA}")

        return psnr_AtoB, psnr_BtoA

    except Exception as e:
        logger.error(f"Error in PSNR calculation for {dataset_name}-{target_index}: {str(e)}")
        return None, None

if __name__ == '__main__':
    MODELS_DIR = 'models/archive'
    COMPRESS_DIR = "/home/jupyter-haoyu/data_202505/r2/r2/test_dataset/block_compress"
    ORIGIN_DIR = "/home/jupyter-haoyu/data_202505/r2/r2/test_dataset/block_origin"
    INTERPOLATE_DIR = "/home/jupyter-haoyu/data_202505/r2/r2/test_dataset/block_interpolate"
    OUTPUT_DIR = './output'

    # 用户指定 model1 和 model2 路径
    specific_model1 = os.path.join(MODELS_DIR, 'epoch_60_model_20250524.pth')
    specific_model2 = os.path.join(MODELS_DIR, 'epoch_60_model_20250605.pth')

    datasets = {
        "redandblack": range(1450, 1460),
        "soldier": range(536, 546)
    }

    predict_results = []

    print("\n进行模型预测...")
    for dataset_name, indices in datasets.items():
        for target_index in indices:
            target_index_str = str(target_index).zfill(4)
            psnr_AtoB, psnr_BtoA = predict(
                model1_path=specific_model1,
                model2_path=specific_model2,
                compress_dir=COMPRESS_DIR,
                interpolate_dir=INTERPOLATE_DIR,
                origin_dir=ORIGIN_DIR,
                output_dir=OUTPUT_DIR,
                target_index=target_index_str,
                dataset_name=dataset_name
            )
            if psnr_AtoB is not None and psnr_BtoA is not None:
                predict_results.append({
                    "Dataset-Target": f"{dataset_name}-{target_index_str}",
                    "PSNR A to B": psnr_AtoB,
                    "PSNR B to A": psnr_BtoA
                })

    df_predict = pd.DataFrame(predict_results)
    print("\nPredict PSNR Results Table:")
    print(df_predict)

    avg_predict_psnr_AtoB = df_predict["PSNR A to B"].mean()
    avg_predict_psnr_BtoA = df_predict["PSNR B to A"].mean()
    print(f"\nAverage Predict PSNR A to B: {avg_predict_psnr_AtoB:.2f} dB")
    print(f"Average Predict PSNR B to A: {avg_predict_psnr_BtoA:.2f} dB")

    logger.info(f"所有 PSNR 结果已保存到 {OUTPUT_DIR} 目录")
