import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn

class LossF(nn.Module):
    def __init__(self, weights = [0.5, 0.5]):
        super(LossF, self).__init__()
        self.weights = weights
        
    def forward(self, pred_mask, pred_edge, gt_mask, gt_edge):
        
        # Mask 类别依然可以用普通的 BCE
        bce_mask = F.binary_cross_entropy(pred_mask, gt_mask)
        
        # Edge 引入简单的 Focal Loss (因为你的模型已经输出了 Sigmoid，需要防 NaN)
        pred_edge_clamp = torch.clamp(pred_edge, 1e-6, 1.0 - 1e-6)
        alpha = 0.75 # 增加正样本(边缘)的权重
        gamma = 2.0
        
        # Focal Loss 展开
        bce_edge_pixel = - (gt_edge * torch.log(pred_edge_clamp) + (1 - gt_edge) * torch.log(1 - pred_edge_clamp))
        pt = torch.exp(-bce_edge_pixel)
        focal_edge = (alpha * gt_edge + (1 - alpha) * (1 - gt_edge)) * ((1 - pt) ** gamma) * bce_edge_pixel
        focal_edge = focal_edge.mean()
        
        def dice_loss(pred, target, smooth=1e-5):
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            intersection = (pred_flat * target_flat).sum()
            # 注意：对于稀疏的Edge，Dice loss分母用平方和有时更稳定，这里保持你的逻辑但调大smooth
            return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        dice_mask = dice_loss(pred_mask, gt_mask)
        dice_edge = dice_loss(pred_edge, gt_edge)
        
        # 分别计算组合Loss
        loss_mask = bce_mask + dice_mask
        loss_edge = focal_edge + dice_edge # edge 替换成了 focal_edge
        
        total_loss = self.weights[0] * loss_mask + self.weights[1] * loss_edge
        
        return total_loss

class LossF_BCE(nn.Module):
    def __init__(self, weights = [0.1, 0.9]):
        super(LossF_BCE, self).__init__()
        self.weights = weights

    def forward(self, pre_mask, pre_edge, mask, edge):

        l1 = F.binary_cross_entropy(pre_mask, mask)
        l2 = F.binary_cross_entropy (pre_edge, edge)

        return self.weights[0] * l1 + self.weights[1] * l2