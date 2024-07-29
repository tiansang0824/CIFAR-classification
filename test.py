# @File：    test.py.py
# @Author:  tiansang0824
# @Time:    2024/7/21 18:05
# @Description: 
#
"""
该文件用于测试模型准确性

"""
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
from Model import Classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取测试数据集
test_dataset = CIFAR10(
    root='./assets/dataset/test/',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

# 获取DataLoader
test_loader = data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True,
    drop_last=False
)

# 从数据中加载模型
model_params = torch.load('./assets/model_params/model_state_dict_9.pth')
classifier = Classifier()
classifier.load_state_dict(model_params)  # 加载模型参数
classifier.to(device)
print(f'>> 模型参数如下：\n{classifier}')

# 配置训练数据
# 定义损失函数
loss_fn = torch.nn.CrossEntropyLoss().to(device)
right_num = 0
test_size = len(test_loader)
total_loss = 0.0

# 开始训练
classifier.eval()
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = classifier(inputs)  # 计算输出
        print(f'outputs.shape: {outputs.shape}; labels.shape: {labels.shape}')
        loss = loss_fn(outputs, labels)  # 计算损失
        total_loss += loss.item()  # 计算总体损失
        right_num += (outputs.argmax() == labels).sum().item()

print(f'平均损失为：{total_loss / test_size}')
print(f'精确度为：{right_num / test_size}')
