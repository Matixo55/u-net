import cv2
import numpy as np

image1 = cv2.imread(r".\original\z23.jpg")
image2 = cv2.imread(r".\original_compressed\z23.jpg")
image3 = cv2.imread(r".\temp\z23.jpg")
difference = cv2.subtract(image1, image3)

difference = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY)


unique, counts = np.unique(difference, return_counts=True)
np.set_printoptions(suppress=True)
print(np.asarray((unique, counts)).T)
difference = (difference / 20) * 255
difference[difference > 255] = 255

cv2.imwrite(r".\xd\test1-3.png", difference)
