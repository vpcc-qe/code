# Video-based Point Cloud Compression Quality Enhancement Using Deep Learning Approach


This repository contains the official implementation for the paper: **"Video-based Point Cloud Compression Quality Enhancement Using Deep Learning Approach"**.

## Prerequisites

Before you begin, ensure you have the following installed:

* **Python:** 3.8.20
* **PyTorch:** 1.9.1+cu111
* **CUDA Version used by PyTorch**: 11.1
* **MinkowskiEngine Version**: 0.5.4



## 1. Initial Ground Truth Generation for BaseNet Training
### Datasets
+ Ensure you have download the MPEG PCC 8i dataset, and use [mpeg-pcc-tmc2](https://github.com/MPEGGroup/mpeg-pcc-tmc2) to generate compressed data series `r1`, `r2`, `r3` from `longdress`, `loot`, `redandblack`, and `soldier`.

+ Run `python basenet_dataset.py`, You will get the following directory: 
 ```diff
 vpcc-qe/
 ├── data_r1/
 │   ├── train_dataset/
 │   │   ├── block_origin/
 │   │   ├── block_compress/
 │   │   ├── block_new_origin/
 │   ├── test_dataset/
 │   │   ├── block_origin/
 │   │   ├── block_compress/
 ├── data_r2/
 ├── data_r3/
 ...
 ```

### Train: 
 
Run `python train.py`, which will generate `epoch_[number]_model.pth` files in the `models` folder.

### Evaluate:
Run `python evaluate.py`, We default choose `epoch_60_model.pth` to evaluate.

## 2. Dynamic Ground Truth Refinement for Enhanced BaseNet Training

### Datasets
Same as above.

### Train:
Same as above, but you need to set `use_vpcc=True` in `data.py`.

### Evaluate:
Same as above.

## 3. Ground Truth Generation for Point Complementation
### Datasets
+ Ensure you have download the `interpolate dataset` and move it into the `train_dataset` and `test_dataset`.
  ```
  vpcc-qe/
  ├── data_r1/
  │   ├── train_dataset/
  │   │   ├── block_origin/
  │   │   ├── block_compress/
  │   │   ├── block_new_origin/
  │   │   ├── block_interpolate/
  │   ├── test_dataset/
  │   │   ├── block_origin/
  │   │   ├── block_compress/
  │   │   ├── block_interpolate/
  ├── data_r2/
  ├── data_r3/
  ...
  ```

+ Run `python interpolatenet_dataset.py`, You will get the following directory: 
  ```
  vpcc-qe/
  ├── data_r1/
  │   ├── train_dataset/
  │   │   ├── block_origin/
  │   │   ├── block_compress/
  │   │   ├── block_new_origin/
  │   │   ├── block_interpolate/
  │   │   ├── block_predict/
  │   │   ├── block_new_predict_origin/
  │   │   ├── block_predict_residual/
  │   │   ├── block_new_predict_residual/
  │   ├── test_dataset/
  │   │   ├── block_origin/
  │   │   ├── block_compress/
  │   │   ├── block_interpolate/
  ├── data_r2/
  ├── data_r3/
  ```


### Train:
- Run `python train_interpolate.py`

### Evaluate:
- Run `python evaluate_interpolate.py`


