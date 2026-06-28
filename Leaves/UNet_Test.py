import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
from torch.utils.data import random_split
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from models.model import UNet
    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
    model = UNet()
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([1024, 1024], interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.367, 0.394, 0.178],
                                std=[0.141, 0.183, 0.103])
    ])

    img = cv2.imread('./figures/0F80BFBF9B128B5FE9DE649C6EFD2945.jpg')
    h, w, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs = transform(img).unsqueeze(0).to(device)

    thre = 0.2
    with torch.no_grad():
        pre_mask, edge = model(imgs)

        pred = (pre_mask > thre).float()   #在这里
        mask = pred[0, 0].cpu().numpy() * 255
        mask = mask.astype(np.uint8)
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('', mask_resized)
        cv2.waitKey(0)
        
        overlay = img.copy()
        overlay[mask_resized > thre] = [255, 0, 0]
        
        result = cv2.addWeighted(img, 0.7, overlay, 1, 0)

        result2 = np.zeros_like(result)
        result2[mask_resized > thre] = img[mask_resized > thre]
        
        cv2.imshow('Original', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imshow('Result', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        cv2.imwrite('asdfdf.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
#======================================================================================================================

    # cap = cv2.VideoCapture(0)
    # print(cap.isOpened())
    # _, frame = cap.read()
    # h, w, _ = frame.shape

    # while(True):
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     img = frame
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = cv2.resize(img, (1024, 1024))
    #     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #     imgs = transform(img).unsqueeze(0).to(device)

    #     with torch.no_grad():
    #         pred, edge = model(imgs)
    #         pred_mask = (pred > 0.2).float()
    #         pred_edge = (edge > 0.2).float()

    #     mask = pred_mask[0, 0].cpu().numpy() * 255
    #     mask = mask.astype(np.uint8)
    #     edge = pred_edge[0, 0].cpu().numpy() * 255
    #     edge = edge.astype(np.uint8)

    #     overlay = img.copy()
    #     overlay[mask > 0] = [0, 0, 255]
    #     result = cv2.addWeighted(img, 0.7, overlay, 1, 0)
    #     result = cv2.resize(result, (w, h))
    #     cv2.imshow('2', frame)
    #     cv2.imshow('1', edge)
    #     cv2.imshow('4', result)
    #     cv2.waitKey(10)

if __name__ == '__main__':
    main()