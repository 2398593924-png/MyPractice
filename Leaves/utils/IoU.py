import numpy as np

def GetIoU(pred, gtmask, thre):
    batch_size = pred.shape[0]
    pred_mask = (pred > thre).float()
    pred_mask = pred_mask.view(batch_size, -1)
    gtmask = gtmask.view(batch_size, -1)

    Intersection = (pred_mask * gtmask).float().sum(dim = 1)
    Union = (pred_mask + gtmask - Intersection).float().sum(dim = 1)

    return ((Intersection + 1e-5) / (Union + 1e-5)).mean()