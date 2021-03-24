# 2021SEUCoronaryArterySegmentation

### 环境配置

Python版本 3.7.7

PyQt5 5.15.1

Mayavi 4.7.2

pytorch 1.8.0

### 运行说明

Mayavi/Codes/Data/background 裁剪后的心脏图片 用于可视化

Mayavi/Codes/Data/image 未裁剪的心脏图片 用于训练

Mayavi/Codes/Data/label 标签 用于训练

Mayavi/Codes/Data/npy 图片的shape文件

Mayavi/Codes/Data/predict 生成的预测图片

**运行时需要保证同一个样本的所有对应文件名相同**

如：

Mayavi/Codes/Data/background/1.raw

Mayavi/Codes/Data/image/1.raw

Mayavi/Codes/Data/label/1.raw

Mayavi/Codes/Data/npy/1.npy

Mayavi/Codes/Data/predict/1.raw

