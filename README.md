# LensePro: Label Noise-Tolerant Prototype-Based Network for Improving Cancer Detection in Prostate Ultrasound with Limited Annotations 

(Submitted to IPCAI2024)

This repository contains the code for the implementation of LensePro. 

## Method
![image](https://github.com/minhto2802/LensePro/assets/26569309/43df69aa-da18-4d20-838c-9fdeeaa45ef8)

## Private Dataset

## Results
![image](https://github.com/minhto2802/LensePro/assets/26569309/a3bec842-af01-4b01-8b3f-bdc7297f1a4d)

## Usage Guideline
- `scripts/run_vicreg_patch.sh` will call `main_pretrain.py` to train a self-supervised learning model using VICReg.
-  `scripts/run_vicreg_patch_evaluate.sh` will call `main_evaluate.py` to perform linear-probing/finetuning on the pretrained model for the classification task.

## Authours
* [**Minh Nguyen Nhat To**](https://github.com/minhto2802)

## Citation
If any part of this code is used, please give appropriate citation to our paper.
