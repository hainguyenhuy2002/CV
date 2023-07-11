# SRGAN-PyTorch

## Overview

This repository contains PyTorch implementations of 3 models for the Super Resolution problem:
1. SRGAN
2. DRRN
3. EDSR

Some information about the project.

## Table of contents

- [Super Resolution](#srgan-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How to Test and Train](#how-to-test-and-train)
        - [Test SRGAN_x4](#test-srgan-x4)
        - [Train SRGAN_x4](#train-srgan-x4)
        - [Resume training SRGAN_x4](#resume-training-srgan-x4)
        - [Test EDSR](#test-edsr)
        - [Train EDSR](#train-edsr)
        - [Test DRRN](#test-drrn)
        - [Train DRRN](#train-drrn)
    - [Results](#results)
    - [Contributing](#contributing)
    - [Credits](#credits)

## Download weights

- [Google Drive](insert link here)

## Download datasets

Contains DIV2K and Set5.

- [Google Drive](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)

Please refer to the README.md in the data directory for instructions on how to create a dataset.

## How Test and Train

*SRGAN*

Both training and testing only need to modify yaml file. 

### Test srgan_x4

```bash
python3 test.py --config_path ./configs/test/SRGAN_X4.yaml
```

### Train srgan_x4

Modify the `./configs/train/SRGAN_X4.yaml` file.

- line 38: `PRETRAINED_G_MODEL` change to `./results/SRResNet_x4-ImageNet/g_last.pth.tar`.

```bash
python3 train_gan.py --config_path ./configs/train/SRGAN_X4.yaml
```

### Resume train srgan_x4

Modify the `./configs/train/SRGAN_X4.yaml` file.

- line 38: `PRETRAINED_G_MODEL` change to `./results/SRResNet_x4-ImageNet/g_last.pth.tar`.
- line 40: `RESUMED_G_MODEL` change to `./samples/SRGAN_x4-ImageNet/g_epoch_xxx.pth.tar`.
- line 41: `RESUMED_D_MODEL` change to `./samples/SRGAN_x4-ImageNet/d_epoch_xxx.pth.tar`.

```bash
python3 train_gan.py --config_path ./configs/train/SRGAN_X4.yaml
```

*EDSR*

### Test edsr
There are 2 ways: by path and by url. These 2 functions can be found inside the predict.py

### Train EDSR
* criterion: nn.L1Loss()
* scale: 2
* epochs: 25
* train_loss: 0
* batch_num: 0
* decay_lr_at: 11, 15 
* decay_lr_to: 0.1

change the wanted argument and run model.py in edsr folder (creating new model or resumed training)

### Test DRRN
There are 2 ways: by path and by url. These 2 functions can be found inside the predict.py

### Train DRRN
* B = 1
* U = 9
* num_features = 32
* lr = 0.1
* weights_file = None
* clip_grad = 0.01
* num_epochs = 50
* num_workers = 2
* seed = 123
* outputs_dir 
* batch_size = 128
* eval_scale = 4

change the wanted argument and run model.py in drrn folder (creating new model or resumed training)

## Result

| Set5 | Scale | SRGAN           | EDSR | DRRN |
|------|-------|-----------------|------|------|
| PSNR |   4   | 29.40 (30.64)   |   -  |   -  |
| PSNR |   3   | 0.8472 (0.8642) |   -  |   -  |
| PSNR |   2   | 0.8472 (0.8642) |   -  |   -  |


## Contributing

This project are proudly made by 5 friends:
1. Nguyen Huy Hai 
2. Bui Thanh Tung 
3. Dao Van Tung 
4. Nguyen Huy Hoang 
5. Duong Vu Tuan Minh

## Credit

### Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

_Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan
Wang, Wenzhe Shi_ <br>

### Enhanced Deep Residual Networks for Single Image Super-Resolution

Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, Kyoung Mu Lee <br>

### Image Super-Resolution via Deep Recursive Residual Network

Ying Tai, Jian Yang, and Xiaoming Liu <br>

**Report link** <br>
Insert link: 
