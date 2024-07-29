# @File：    train.py
# @Author:  tiansang0824
# @Time:    2024/7/21 16:59
# @Description: 
#
"""
该文件用于训练模型。

"""
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Model import Classifier

# 设定训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取训练数据集
train_dataset = datasets.CIFAR10(
    root='./assets/dataset/train/',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# 创建DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    drop_last=False
)

# 创建一个模型实例
classifier = Classifier()
classifier.to(device)  # 设定训练设备

# 指定损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 定义优化器
learning_rate = 0.001
optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)  # 梯度下降优化

# 设置训练网络的参数
total_train_step = 0  # 总训练次数，该参数用于TensorBoard总结的时候使用。
epochs = 10  # 训练50轮

# 创建TensorBoard-writer
writer = SummaryWriter(log_dir='./assets/tensorboard_logs/train_logs/')

# 开始训练模型
classifier.train()
for i in range(epochs):
    print(f'>> epoch {i + 1} started...')

    # 开始训练步骤
    for data in train_loader:
        # 获取数据
        inputs, labels = data[0].to(device), data[1].to(device)
        # 计算y_hat
        outputs = classifier(inputs)
        # 开始训练
        loss = loss_fn(outputs, labels)  # 计算损失
        optimizer.zero_grad()  # 清除梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 优化参数
        # 修改全局记录
        total_train_step += 1  # 训练次数+1
        # 保存数据
        if total_train_step % 100 == 0:
            print(f'\ttotal train step: {total_train_step}, \tloss: {loss.item()}')  # 输出训练步骤和对应损失
            writer.add_scalar('model_loss_history', loss.item(), total_train_step)  # 保存日志，用于TensorBoard

    # 保存模型参数
    torch.save(classifier.state_dict(), f'./assets/model_params/model_state_dict_{i}.pth')

print(f'\n\n\n>> 训练完成!\n\n\n')
