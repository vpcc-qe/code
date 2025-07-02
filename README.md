# [WIP]: Video-based Point Cloud Compression Quality Enhancement Using Deep Learning Approach


This repository contains the official implementation for the paper: **"Video-based Point Cloud Compression Quality Enhancement Using Deep Learning Approach"**.

## Prerequisites

Before you begin, ensure you have the following installed:

* **Python:** 3.8.20
* **PyTorch:** 1.9.1+cu111
* **CUDA Version used by PyTorch**: 11.1
* **MinkowskiEngine Version**: 0.5.4


## 1. Initial Ground Truth Generation for BaseNet Training
### Datasets
Ensure you have download the MPEG PCC 8i dataset, and place it under the `data/` directory, expected directory structure:
```
vpcc-qe/
├── data/
│   ├── 8i/
│   │   ├── 8iVFBv2/
│   │   │   ├── longdress
│   │   │   ├── loot
│   │   │   ├── redandblack
│   │   │   ├── soldier
```

Next, use `mpeg-pcc-tmc2` to generate compressed data series `r1`, `r2`, `r3` from `longdress`, `loot`, `redandblack`, and `soldier`.

Next, cut the compressed data into blocks by running `python cut_block.py`

After cutting blocks, find `new_origin`. You need to use the kdtree method from `mpeg-pcc-dmetric-0.13.05` to generate the `block_compress` folder.

### Train: 
 
Run `python train.py`, which will generate `epoch_[number]_model.pth` files in the `models` folder.

## 2. Dynamic Ground Truth Refinement for Enhanced BaseNet Training

### Datasets
Same as above
### Train:

Same as above, but you need to modify `use_vpcc=True`

## 3. Ground Truth Generation for Point Complementation

### Datasets

- Download the corresponding `block_interpolates` files for the `r1`, `r2`, `r3` series and place them in the corresponding folders
- Select the model file from step 2, run `python gen_predict.py` to get the corresponding `block_predict` from `block_compress` through the model file
- Run `python gen_origin.py`, pass `block_predict` and `block_origin` as parameters to get `block_new_predict_origin`
- Run `python gen_residual.py`, pass `block_predict` and `block_new_predict_origin` as parameters to get `block_predict_residual`
- Run `python gen_origin.py`, pass `block_predict_residual` and `block_interpolate` as parameters to get `block_new_predict_residual`

### Train:

- Run `python train.py`, pass `block_interpolate` and `block_new_predict_residual` as parameters for training.

### Evaluate:
- Run `python `


