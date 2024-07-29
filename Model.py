# @File：    Model.py
# @Author:  tiansang0824
# @Time:    2024/7/21 16:59
# @Description: 
#
"""
定义神经网络模型

本文件使用PyTorch创建了一个分类用神经网络，

该网络结构图可参考`project_root/assets/model_structure.jpeg`文件，
或者参考`readme.md`文件介绍。

训练和测试集均采用了CAFAR10数据集，数据集具体情况可以参考：https://www.cs.toronto.edu/~kriz/cifar.html

"""
import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义网络模型
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)
        # return nn.Softmax(self.model(x))

