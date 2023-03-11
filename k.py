import collections

import numpy as np
from PIL import Image, ImageChops
from matplotlib import pyplot as plt

for a in range(1, 8):
    im = Image.open(rf"C:\Users\Mateusz\Desktop\2\{a}-100.jpg")
    im2 = Image.open(rf"C:\Users\Mateusz\Desktop\2\{a}-25.jpg")

    x = np.array(ImageChops.difference(im, im2))
    # x = (x<7) * 1
    # x = rgb2gray(x)
    # print(x)
    xd = x > 10

    unique, counts = np.unique(xd, return_counts=True)
    print(dict(zip(unique, counts)))
    k = np.where(xd, im, im2)
    Image.fromarray(k).save(rf"C:\Users\Mateusz\Desktop\{a}.jpg", quality=100)
    # plt.imshow(k)
    # plt.show()
    # x[x!=1] = 0
    # # x*= 5
    # plt.imshow(x, cmap="binary",vmin=0, vmax=1)
    # plt.show()
