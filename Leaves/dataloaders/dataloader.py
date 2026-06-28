import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode

GTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([1024, 1024], interpolation=InterpolationMode.NEAREST),
    transforms.ColorJitter(brightness=0.3,
                            contrast=0.3,
                            saturation=0.3,
                            hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.367, 0.394, 0.178],
                            std=[0.141, 0.183, 0.103])
])

class Ara2013(Dataset):
    def __init__(self, path):
        self.path = path + '/Ara2013/'
        self.num = 27
        self.transform_input = GTransform
        self.transform_label = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([1024, 1024], interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        idx += 1
        img = cv2.imread(self.path + f"ara2013_tray{idx:0{2}d}_rgb.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform_input(img)

        label = cv2.imread(self.path + f"ara2013_tray{idx:0{2}d}_fg.png")
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = self.transform_label(label)

        edge = torch.zeros_like(label)

        return img, label, edge

class Ara2012(Dataset):
    def __init__(self, path):
        self.path = path
        self.num = 40
        self.transform_input = GTransform

        self.transform_label = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([1024, 1024], interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        self.samples = []
        for i in range(1, self.num + 1):
            if i in range(1, 121):
                path = self.path + '/Ara2012/'
                path_img = path + f"ara2012_plant{i:0{3}d}_rgb.png"
                path_mask = path + f"ara2012_plant{i:0{3}d}_label.png"
                path_edge = path + f"ara2012_plant{i:0{3}d}_boundaries.png"
            self.samples.append({
                'img': path_img,
                'mask': path_mask,
                'edge': path_edge
            })

    def __len__(self):
        return self.num
    
    def __getitem__(self, idx):
        data = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(data['img']), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(data['mask']), cv2.COLOR_BGR2GRAY)
        mask[mask > 0.2] = 255
        edge = cv2.cvtColor(cv2.imread(data['edge']), cv2.COLOR_BGR2GRAY)
        edge[edge > 0.2] = 255

        # kernel = np.ones((3, 3), np.uint8)
        # edge = cv2.dilate(edge, kernel, iterations=1)

        img = self.transform_input(img)
        mask, edge = self.transform_label(mask), self.transform_label(edge)

        return img, mask, edge
    
class Tobacco(Dataset):
    def __init__(self, path):
        self.path = path
        self.num = 27
        self.transform_input = GTransform

        self.transform_label = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([1024, 1024], interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return self.num
    
    def __getitem__(self, index):
        index += 1
        path = self.path + '/Tobacco/'
        path_img = path + f"tobacco_plant{index:0{3}d}_rgb.png"
        path_mask = path + f"tobacco_plant{index:0{3}d}_label.png"
        path_edge = path + f"tobacco_plant{index:0{3}d}_boundaries.png"

        img = cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(path_mask), cv2.COLOR_BGR2GRAY)
        mask[mask > 0.5] = 255
        edge = cv2.cvtColor(cv2.imread(path_edge), cv2.COLOR_BGR2GRAY)
        edge[edge > 0.5] = 255

        kernel = np.ones((3, 3), np.uint8) # 可以试试 (5, 5) 如果依然学不到
        edge = cv2.dilate(edge, kernel, iterations=1)

        img = self.transform_input(img)
        mask, edge = self.transform_label(mask), self.transform_label(edge)

        return img, mask, edge

class Ara2013_Canon(Dataset):
    def __init__(self, path):
        self.path = path
        self.num = 62
        self.transform_input = GTransform

        self.transform_label = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([1024, 1024], interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        idx += 1
        img = cv2.imread(self.path + f"ara2013_tray{idx:0{2}d}_rgb.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img[img > 0.5] = 255
        img = self.transform_input(img)

        label = cv2.imread(self.path + f"ara2013_tray{idx:0{2}d}_fg.png")
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label[label > 0.5] = 255
        label = self.transform_label(label)

        return img, label
    
def GetLoader(path, batch_size, type, num_workers = 2):

    match type:
        case 1:
            dataset = Ara2013(path)
        case 2:
            dataset = Ara2012(path)
        case 3:
            dataset = Tobacco(path)
    
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    TrainLoader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers = num_workers)
    TestLoader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers = num_workers)

    return TrainLoader, TestLoader