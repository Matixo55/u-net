import cv2
from numpy import amax
import numpy as np

# load images
image1 = cv2.imread(r".\original\z23.jpg")
image2 = cv2.imread(r".\original_compressed\z23.jpg")
image3 = cv2.imread(r".\temp\z23.jpg")
# compute difference
difference = cv2.subtract(image1, image3)

difference = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY)


unique, counts = np.unique(difference, return_counts=True)
np.set_printoptions(suppress=True)
print(np.asarray((unique, counts)).T)
difference = (difference / 20) * 255
prog = 100
difference[difference > 255] = 255
# difference[difference <= prog] = 0
# _,alpha = cv2.threshold(difference,0,255,cv2.THRESH_BINARY)

# print(alpha.shape)
# img_3gray = cv2.merge((difference,difference,difference,alpha), 4)

# store images
cv2.imwrite(r".\xd\test1-3.png", difference)
