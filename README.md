# Pytorch框架学习笔记

## 开发环境配置
1、操作系统：win10
2、CPU：Core i5-8265U @ 1.60GHz 四核
3、Anaconda3-4.11.0-Windows-x86_64

安装包要和本地的CPU架构一致
```bash
# 查看本机CPU架构
lscpu
```

运用conda管理python环境
```bash
# 创建虚拟环境
conda create -n pytorch python=3.9

# 使用conda安装
conda install pytorch -c pytorch
conda install torchvision -c pytorch
conda install matplotlib -c pytorch
conda install pandas -c pytorch

# 使用python的pip安装
conda activate pytorch
pip install torchvision
```


## 编写深度学习模型基本流程
在我们要用pytorch构建自己的深度学习模型的时候，基本上都是下面这个流程步骤，写在这里让一些新手童鞋学习的时候有一个大局感觉，无论是从自己写，还是阅读他人代码，按照这个步骤思想（默念4大步骤，找数据定义、找model定义、(找损失函数、优化器定义)，主循环代码逻辑），直接去找对应的代码块，会简单很多。

基本步骤思想
分为四大步骤：
1、输入处理模块（X输入数据，变成网络能够处理的Tensor类型）
2、模型构建模块（主要负责从输入的数据，得到预测的y^,这就是我们常说的前向过程）


## 核心类的使用
### datasets&dataloader类的使用

### torch.utils.data.Subset类的使用
划分数据集

代码样例
```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
from torch.utils.data import Subset

train_set = torchvision.datasets.MNIST(root="./data",train=True,transform=transforms.ToTensor(),download=True)
train_set_A=Subset(train_set,range(0,1000))
train_set_B=Subset(train_set,range(1000,2000))
train_set_C=Subset(train_set,range(2000,3000))
train_loader_A = dataloader.DataLoader(dataset=train_set_A,batch_size=1000,shuffle=False)
train_loader_B = dataloader.DataLoader(dataset=train_set_B,batch_size=1000,shuffle=False)
train_loader_C = dataloader.DataLoader(dataset=train_set_C,batch_size=1000,shuffle=False)
test_set = torchvision.datasets.MNIST(root="./data",train=False,transform=transforms.ToTensor(),download=True)
test_set=Subset(test_set,range(0,2000))
test_loader = dataloader.DataLoader(dataset=test_set,shuffle=True)
```

