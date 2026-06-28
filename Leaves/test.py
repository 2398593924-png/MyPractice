import cv2
import numpy as np
import torch
from dataloaders.dataloader import GetLoader
from torch.utils.data import DataLoader

def main():

    TrainLoader, _ = GetLoader('./datasets', 1, 1, 1)
    img, label, _ = next(iter(TrainLoader))
    with torch.no_grad():
        img = img.squeeze(0).detach().numpy()
        label = label.squeeze(0).detach().numpy()

    img = np.transpose(img, (1, 2, 0))
    label = np.transpose(label, (1, 2, 0))  #(B, C, H, W) -> (H, W, C)

    h, w, _ = img.shape
    thre = 2

    cv2.resize(label, (h//thre, w//thre))
    cv2.imshow('original', img)
    cv2.imshow('label', label)

    cv2.waitKey(0)

if __name__ == '__main__':
    main()
