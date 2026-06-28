import glob
import re
import os
import cv2
import numpy as np


mean = 0
std = 0
n = 147

# Ara2012:
# Mean:[0.384, 0.422, 0.161]  Std:[0.132, 0.181, 0.085]
# Ara2013-Canon
# Mean:[0.261, 0.279, 0.142]  Std:[0.101, 0.178, 0.051]
# Tobacco
# Mean:[0.104, 0.126, 0.109]  Std:[0.118, 0.138, 0.129]
# Ara2013
# Mean:[0.287, 0.272, 0.255]  Std:[0.181, 0.195, 0.183]

# Global
# Mean:[0.284, 0.308, 0.159]  Std:[0.128, 0.183, 0.094]

# Ara2012 + Ara2013
# Mean:[0.367, 0.394, 0.178]  Std:[0.141, 0.183, 0.103]

for i in range(1, 375):
    # print(path + f"ara2013_tray{i:0{2}d}_rgb.png")
    # 1~120 Ara2012
    # 121~285 Ara2013-Canon
    # 286~347 Tobacco
    # 348~374 Ara2013
    if i in range(1, 121):
        path = './datasets/Ara2012/'
        path = path + f"ara2012_plant{i:0{3}d}_rgb.png"
    if i in range(121, 286):
        continue
        path = './datasets/Ara2013-Canon/'
        path = path + f"ara2013_plant{(i - 120):0{3}d}_rgb.png"
    if i in range(286, 348):
        continue
        path = './datasets/Tobacco/'
        path = path + f"tobacco_plant{(i - 285):0{3}d}_rgb.png"
    if i in range(348, 375):
        path = './datasets/Ara2013/'
        path = path + f"ara2013_tray{(i - 347):0{2}d}_rgb.png"

    if os.path.exists(path):
        print(f"Processing...{i}")
        img = cv2.imread(path)
        mean += img.mean(axis=(0, 1)) / 255.0
        std += img.std(axis=(0, 1)) / 255.0
        
print(f"Mean: {mean / n}")
print(f"Std: {std / n}")