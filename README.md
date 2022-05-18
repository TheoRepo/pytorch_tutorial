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
3、定义代价函数和优化器模块（注意，前向过程只会得到模型预测的结果，并不会自动求导和更新，是由这个模块进行处理）
4、构建训练过程（迭代训练过程）

## 基本代码块
### 数据处理
对于数据处理，最为简单的方式就是将数据组织成为一个。但是训练需要用到mini-batch,直接组织成Tensor不便于我们操作，pytorch为我们提供了Dataset和Dataloader两个类方便我们构建

继承Dataset类需要override以下方法
```python
from torch.utils.data import Dataset
class trainDataset(Dataset):
	def __init__(self):
		# constructor
	
	def __getitem__(self, index):
		# 获得index号的数据和标签
		
	def __len__(self)：
		# 获得数据量
```
迭代DataLoader获取数据
```python
dataLoader = DataLoader(dataset, shuffle=True, batch_size=32)
for i, data in enumerate(dataLoader, 0):
	x, y = data
	# 同时获得数据和标签
```

### 模型构建
所有模型都需要继承torch.nn.Module，需要实现以下方法：
其中forward()方法是前向传播的过程，在实现模型时，我们不需要考虑反向传播
```python
class MyModel(torch.nn.Module):
	def __init__(self):
		super(MyModel, self).__init__()
		# to do
	
	def forward(self, x):
		# to do
		
model = MyModel()
```

### 定义代码函数和优化器
```python
criterion = torch.nn.BCELoss(reduction='sum') #代价函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps= 1e-08, weight_decay=0, amsgrad=False) # 优化器
```

### 构建训练过程
pytorch的训练循环大致如下：
```python
def train(epoch): # 一个epoc的训练
	for i, data in enumerate(dataLoader, 0):
		x, y = data # 取出minibatch数据和标签
		y_pred = model(x) # 前向传播
		loss = criterion(y_pred, y) # 计算代价函数
		optimizer.zero_grad() # 清零梯度准备计算
		loss.backward() # 反向传播
		optimizer.step() # 更新训练参数
```

## 核心类的使用
### datasets&dataloader类的使用
实现transform函数后，即可自定义处理，不同格式的原始数据

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

