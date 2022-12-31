
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage
import numpy as np
import pandas as pd
import cv2
import torchvision.transforms as transforms
from PIL import Image
import os
import pathlib

classes_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

class GetDataset(Dataset):
    def __init__(self, images, labels, transform=None, num_classes=7):
        self.images = images
        self.labels = labels
        self.transform = transform

        # 统计每个类出现了多少次
        self.num_class_list = self.get_num_class_list(num_classes)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = np.array(self.images[idx])

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx]).type(torch.long)

        return img, label
    
    def get_num_class_list(self, num_classes):
        num_list = [0] * num_classes
        print("Weight List has been produced")
        for label in self.labels:
            num_list[int(label)] += 1
        return num_list

def get_data_path(path):
    # 获取数据集路径
    if (os.path.isabs(path)):
        return path
    else:
        data_path = str(os.path.join(pathlib.Path(__file__).parent.resolve(), 'data'))
        return str(os.path.join(data_path, path))

def load_data_from_csv(path):
    # 从csv中获取数据
    data = pd.read_csv(path)
    return data

def csv_data_to_numpy(data):
    # DataFrame数据处理
    # csv读到的数据转成np数组

    # fer2013的数据是48×48的灰度图像
    image = np.zeros(shape=(len(data), 48, 48), dtype=int)
    label = np.zeros(shape=(len(data)), dtype=int)

    for i, row in enumerate(data.index):
        # pixels 列
        pixels = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ').reshape(48,48).astype(np.uint8)

        image[i] = pixels

        emotion = int(data.loc[row, 'emotion'])

        label[i] = emotion

    return image, label

def create_train_data_transform(input_size=40):
    # 训练数据转化，数据增强
    return transforms.Compose([
            # 灰度
            transforms.Grayscale(),
            # 随机裁剪
            transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
            # 随机颜色抖动
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            # 随机仿射变换
            transforms.RandomApply(
                [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            # 随机水平翻转
            transforms.RandomHorizontalFlip(),
            # 随机旋转
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            # 5-crop
            transforms.FiveCrop(input_size),
            # 转化成 tensor，自动归一化到[0,1]，但是没有调整分布
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(crop) for crop in crops])),
            # 归一化
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.Normalize(mean=(0.5077,), std=(0.2544,))(t) for t in tensors])),
            # 随机擦除
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.RandomErasing()(t) for t in tensors]))
        ])

def create_test_data_transform(input_size=40):
    # 测试和验证数据转化，数据增强
    return transforms.Compose([
            # 灰度
            transforms.Grayscale(),
            # 10-crop
            transforms.TenCrop(input_size),
            # 转化成 tensor
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(crop) for crop in crops])),
            # 归一化
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.Normalize(mean=(0.5077,), std=(0.2544,))(t) for t in tensors]))
        ])

def print_dataloader(dataloader, mode):
    for X, y in dataloader:
        # batch_size, ncrop, c, h, w = X.shape
        # img = X.view(-1, c, h, w)
        # show = ToPILImage() # 可以把Tensor转成Image，方便可视化

        # for i in range(10):
        #     show(img[i]).save(f'./image/transform/{i:>2d}.png')
        print(f'\t{mode} data X [BatchSize, NCrop, C, H, W]: \n\t\tshape={X.shape}, \n\t\tdtype={X.dtype}')
        print(f'\t{mode} data Y: \n\t\tshape={y.shape}, \n\t\tdtype={y.dtype}')
        break

    print(f"The num of each class: {dataloader.dataset.num_class_list}")

def get_dataloader(path='fer2013.csv', batch_size=64, input_size=40, mode='Train'):
    # 获取 dataloader
    # usage 训练（Training）、公有测试（PublicTest）、私有测试（PrivateTest）
    path = get_data_path(path)
    print(f'Loading thedataset from: {path}')

    fer2013 = load_data_from_csv(path)

    # 根据用途划分数据集 训练（Training）、验证（PublicTest）、测试（PrivateTest）
    if mode == 'Train':
        transform = create_train_data_transform(input_size)
        usage = 'Training'
    elif mode == 'Val':
        transform = create_test_data_transform(input_size)
        usage = 'PublicTest'
    elif mode == 'Test':
        transform = create_test_data_transform(input_size)
        usage = 'PrivateTest'
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        usage = 'Training'

    X, y = csv_data_to_numpy(fer2013[fer2013['Usage'] == usage])

    dataset = GetDataset(X, y, transform)

    # 读的时候打乱顺序shuffle=True
    data_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print_dataloader(data_dataloader, mode)

    return data_dataloader

def getStat():
    '''
    计算训练集的均值和方差
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    train_loader = get_dataloader(mode='Other')
    print(len(train_loader))
    mean = torch.zeros(1)
    std = torch.zeros(1)
    for X, _ in train_loader:
        for d in range(1):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_loader))
    std.div_(len(train_loader))
    return list(mean.numpy()), list(std.numpy())

def main():
    get_dataloader()
    
if __name__ == "__main__":
    main()