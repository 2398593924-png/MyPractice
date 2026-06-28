import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channel, ratio):
        super(SEBlock, self).__init__()
        self.in_channel = in_channel
        self.ratio = ratio
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_channel, self.in_channel // self.ratio, 1),
            nn.ReLU(),
            nn.Conv2d(self.in_channel // self.ratio, self.in_channel, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        if (self.in_channel // self.ratio >= 1):
            w = self.se(x)
        return w * x
    
class PyramidSpatialAttention(nn.Module):
    def __init__(self, kernel_sizes=[3, 5, 7]):
        super(PyramidSpatialAttention, self).__init__()
        
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            self.convs.append(
                nn.Conv2d(2, 1, k, padding=k//2, bias=False)
            )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_feat = torch.cat([avg_out, max_out], dim=1)
        
        attention_maps = []
        for conv in self.convs:
            att = conv(spatial_feat)
            attention_maps.append(att)
        
        attention = torch.stack(attention_maps, dim=0).mean(dim=0)
        attention = self.sigmoid(attention)
        
        return x * attention
    
class DownScale(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownScale, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.down = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, 3, padding=1, stride=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1, stride=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.down(x)
    
class UpScale(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpScale, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.se = SEBlock(out_channel, ratio=4)
        self.attention = PyramidSpatialAttention()

        self.conv1 = nn.ConvTranspose2d(self.in_channel, self.out_channel, 2, stride=2)
        self.conv2 = nn.Conv2d(self.out_channel * 2, self.out_channel, 3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1, stride=1)

        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x, x_last):
        x = self.conv1(x)
        x_last = self.se(x_last)
        x = torch.concat([x, x_last], dim=1)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.attention(x)
        x = self.conv3(x)
        x = F.relu(x)
        return x
    

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(2)

        self.en1 = DownScale(3, 64)
        self.en2 = DownScale(64, 128)
        self.en3 = DownScale(128, 256)
        self.en4 = DownScale(256, 512)
        self.en5 = DownScale(512, 1024)

        self.de1 = UpScale(1024, 512)
        self.de2 = UpScale(512, 256)
        self.de3 = UpScale(256, 128)
        self.de4 = UpScale(128, 64)

        self.mask_head = nn.Sequential(
            nn.Conv2d(64, 1, 1)
        )

        self.edge_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x1 = self.en1(x)
        x2 = self.en2(self.pool(x1))
        x3 = self.en3(self.pool(x2))
        x4 = self.en4(self.pool(x3))
        x5 = self.en5(self.pool(x4))

        x = self.de1(x5, x4)
        x = self.de2(x, x3)
        x = self.de3(x, x2)
        x = self.de4(x, x1)
        
        
        # mask = self.mask_head(x)
        edge = self.edge_head(x)
        # center = self.center_head(x)

        return edge