# Video-based Point Cloud Compression Quality Enhancement Using Deep Learning Approach


This repository contains the official implementation for the paper: **"Video-based Point Cloud Compression Quality Enhancement Using Deep Learning Approach"**.

## Prerequisites

Before you begin, ensure you have the following installed:

* **Python:** 3.8.20
* **PyTorch:** 1.9.1+cu111
* **CUDA Version used by PyTorch**: 11.1
* **MinkowskiEngine Version**: 0.5.4
* **Dataset**: Download the dataset from [Onedrive](https://mailouhkedu-my.sharepoint.com/:u:/g/personal/s1360912_live_hkmu_edu_hk/EfXAKuDoG2hAhw8JnkeixmABKRMl6RylxNIl8oIVggwrjQ?e=zn7eaa)



## Training the BaseNet

```bash
python train_basenet.py \
    --dataset_root [path/to/your/dataset] \
    --output_dir ./checkpoints/basenet 
```
- `--dataset_root`: Path to the root directory of your training dataset.
- `--output_dir`: Directory where model checkpoints (e.g., ⁠epoch_60_model.pth) and logs will be saved.

## Evaluating the BaseNet

```bash
python evaluate.py \
    --model_path ./checkpoints/basenet/epoch_60_model.pth \
    --dataset_root [path/to/your/test_dataset] \
    --output_dir ./results/basenet_eval
```
- `--model_path`: Path to the specific model checkpoint you want to evaluate. 
- `--dataset_root`: Path to the root directory of your training dataset.
- `--output_dir`: Directory where model checkpoints (e.g., ⁠epoch_60_model.pth) and logs will be saved.


## Training the Enhanced BaseNet

```bash
python train_basenet.py \
    --dataset_root [path/to/your/dataset] \
    --output_dir ./checkpoints/enhanced_basenet \
    --use_dynamic
```
- `--use_dynamic`: Enable dynamic ground truth

## Evaluating the Enhanced BaseNet

```bash
python evaluate.py \
    --model_path ./checkpoints/enhanced_basenet/epoch_60_model.pth \
    --dataset_root [path/to/your/test_dataset] \
    --output_dir ./results/enhanced_basenet_eval
```


## Training the InterpolateNet

```bash
python train.py \
    --dataset_root [path/to/your/dataset] \
    --output_dir ./checkpoints/interpolatenet
    --use_interpolate
```

## Evaluating the InterpolateNet
```bash
python evaluate.py \
    --model_path ./checkpoints/basenet/epoch_60_model.pth \
    --interpolate_model_path ./checkpoints/interpolatenet/epoch_60_model.pth \
    --dataset_root [path/to/your/test_dataset] \
    --output_dir ./results/interpolatenet
```
- `--model_path`: Path to the  BaseNet model or Enhanced BaseNet model checkpoint you want to evaluate. 
- `--interpolate_model_path`: Path to the InterpolateNet model checkpoint you want to evaluate. 


