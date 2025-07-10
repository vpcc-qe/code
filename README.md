# Video-based Point Cloud Compression Quality Enhancement Using Deep Learning Approach


This repository contains the official implementation for the paper: **"Video-based Point Cloud Compression Quality Enhancement Using Deep Learning Approach"**.

## Prerequisites

Before you begin, ensure you have the following installed:

* **Python:** 3.8.20
* **PyTorch:** 1.9.1+cu111
* **CUDA Version used by PyTorch**: 11.1
* **MinkowskiEngine Version**: 0.5.4
* **block dataset**: Download the block dataset from [Onedrive](https://mailouhkedu-my.sharepoint.com/:u:/g/personal/s1360912_live_hkmu_edu_hk/EfXAKuDoG2hAhw8JnkeixmABKRMl6RylxNIl8oIVggwrjQ?e=zn7eaa)



## Training the BaseNet

```bash
python src/train.py --dataset=YOUR_DATASET
```
- The `dataset` are cut blocked point cloud folders.

## Evaluating the BaseNet

```bash
python src/evaluate.py \
--dataset=YOUR_TEST_DATASET \
--model=YOUR_MODEL_PATH \
--vpcc=YOUR_VPCC_TOOL_PATH \
--origin_path=YOUR_ORIGIN_FILE_PATH \
--compress_path=YOUR_COMPRESS_FILE_PATH
```
- The `origin_path` and `compress_path` are not cut blocked point cloud folders.


## Training the Enhanced BaseNet


```bash
python src/train.py --dataset=YOUR_DATASET --dynamic
```

## Evaluating the Enhanced BaseNet

Same as Evaluating the BaseNet.


## Training the InterpolateNet
```bash
python src/train.py --dataset=YOUR_DATASET --interpolate
```


## Evaluating the InterpolateNet

```bash
python src/evaluate.py \
--dataset=YOUR_TEST_DATASET \
--model=YOUR_MODEL_PATH \
--interpolate_model=YOUR_INTERPOLATE_MODEL_PATH \
--vpcc=YOUR_VPCC_TOOL_PATH \
--origin_path=YOUR_ORIGIN_FILE_PATH \
--compress_path=YOUR_COMPRESS_FILE_PATH
```

## Visualization
```bash
f3d compress.ply --output=compress.png --background-color=#EAEAEA --camera-position=-294.714,886.272,564.6 --camera-focal-point=195.834,874.175,244.685 --camera-view-up=0.0172984,0.999787,-0.0112812 --camera-view-angle=33.1667
```

