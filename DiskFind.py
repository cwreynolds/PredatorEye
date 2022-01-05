# -*- coding: utf-8 -*-

# make change from inside notebook

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 20220104 this is wrong!
# It duplicates the definition in Find_Conspicuous_Disk.ipynb
# But doing it temporarily to help test this file DiskFind.py
fcd_image_size = 1024
fcd_disk_size = 201


# Draw a training image on the log. First arg is either a 24 bit RGB pixel
# representation as read from file, or the rescaled 3xfloat used internally.
# Optionally draw crosshairs to show center of disk.
def draw_image(rgb_pixel_tensor, center=(0,0)):
    i24bit = []

    # 20211218
    # if (rgb_pixel_tensor.dtype == np.double):

    # 20211221
    # print('rgb_pixel_tensor.dtype =', rgb_pixel_tensor.dtype)
    # print('center =', center)

    # if (rgb_pixel_tensor.dtype == np.float32):
    if ((rgb_pixel_tensor.dtype == np.float32) or
        (rgb_pixel_tensor.dtype == np.float32)):
        unscaled_pixels = np.interp(rgb_pixel_tensor, [0, 1], [0, 255])
        i24bit = Image.fromarray(unscaled_pixels.astype('uint8'), mode='RGB')
    else:
        i24bit = Image.fromarray(rgb_pixel_tensor)
    plt.imshow(i24bit)

    # 20211221
    # if (center != (0,0)):
    #     draw_crosshairs(center)
    if ((center[0] != 0) or (center[1] != 0)):
        draw_crosshairs(center)

    plt.show()

# Draw crosshairs to indicate disk position (label or estimate).
def draw_crosshairs(center):
    m = fcd_image_size - 1       # max image coordinate
    s = fcd_disk_size * 1.2 / 2  # gap size (radius)
    h = center[0] * m            # center x in pixels
    v = center[1] * m            # center y in pixels
    plt.hlines(v, 0, max(0, h - s), color="black")
    plt.hlines(v, min(m, h + s), m, color="black")
    plt.vlines(h, 0, max(0, v - s), color="white")
    plt.vlines(h, min(m, v + s), m, color="white")
