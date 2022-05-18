import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


# loading a dataset
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


# Iterating and visualizing the dataset
# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     # 对数据的维度进行压缩
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()


# creating a custom dataset for your files
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    # ini函数实例化Dataset对象时运行一次
    # CV的初始化，包含，图像、注释文件和两种转换的目录
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # len函数返回我们数据集中的样本数
    def __len__(self):
        return len(self.img_labels)

    # getitem函数从给定义索引idx的数据集中，加载并返回一个样本。
    # 根据索引，它识别图像在磁盘上的位置，使用read_image将其转换为张量
    # 从self.img_labels中的csv数据中检索相应的标签，调用它们的转换函数transform
    # 返回张量图像和元组中的相应标签
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


from torch.utils.data import DataLoader

# dataloader是一个迭代器，它通过一个简单的API为我们抽象了这种复杂性
# 检索我们数据集的特征并一次标记一个样本。
# 在训练模型时，我们通常希望以“小批量”的形式传递样本，在每个epoch重新洗牌以减少模型过度拟合
# 并使用Python的多处理来加速数据检索
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# iterate through the dataloader
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")