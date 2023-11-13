import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import os
import csv
import numpy as np
import cv2


# 从csv文件中读取训练集和测试集路径信息
def load_data_csv(root, csv_name):
    image_list, label_list = [], []
    with open(os.path.join(root, csv_name)) as f:
        reader = csv.reader(f)
        for row in reader:
            img, label = row
            image_list.append(img)
            label_list.append(label)
    # 返回图片路径list和标签list
    return image_list, label_list


# 合成图像
def composite_image(img, mask):
    # 读入原始图片
    origin = np.array(img)
    # 随机生成噪声图片
    noisy = np.random.randint(1, 100, size=[256, 256, 3])

    mask = np.array(mask)
    mask = mask[:, :, np.newaxis]
    mask = mask.repeat(3, axis=2)
    # 合成图片
    composite = origin * (1 - mask) + noisy * mask
    composite = cv2.blur(composite, (3, 3))

    return Image.fromarray(np.uint8(composite)).convert('RGB')


# 自定义数据集
class My_train_Dataset(Dataset):
    def __init__(self, data_path, csv_name):
        self.data_list, self.mask_list = load_data_csv(data_path, csv_name)
        self.data_transform = transforms.Compose([
            transforms.RandomAdjustSharpness(sharpness_factor=4, p=0.5),
            transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        img = Image.open(self.data_list[index]).convert('RGB')
        mask = Image.open(self.mask_list[index]).convert('1')
        img_compose = composite_image(img, mask)
        img_compose = self.data_transform(img_compose)
        img = self.mask_transform(img)
        return img_compose, img, self.data_list[index], self.mask_list[index]

    def __len__(self):
        return len(self.data_list)


class My_test_Dataset(Dataset):
    def __init__(self, data_path, csv_name):
        self.data_list, self.mask_list = load_data_csv(data_path, csv_name)
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = Image.open(self.data_list[index]).convert('RGB')
        img = self.data_transform(img)
        mask = Image.open(self.mask_list[index]).convert('1')
        mask = self.mask_transform(mask)
        return img, mask, self.data_list[index], self.mask_list[index]

    def __len__(self):
        return len(self.data_list)