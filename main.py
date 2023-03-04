import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.models import Model
from PIL import ImageChops

from model import unet
from data import *
import numpy as np

# im = Image.open(r'C:\Users\Mateusz\Desktop\dataset_bad\11.jpg')
#
# r, g, b = im.split()
# r= r.crop((92,92,r.width-92,r.height-92))
# plt.imshow(r)
# plt.show()
#
# im2 = Image.open(r'C:\Users\Mateusz\Desktop\dataset_100\11.jpg')
#
#
# r2, g, b = im2.split()
# diff = ImageChops.difference(r, r2)
# x = np.array(diff)
# x = (x>20) *1
# print(x)
# plt.imshow(x, cmap='gray')
# plt.show()
# plt.imshow(r2)
# plt.show()
# im = Image.open(r'C:\Users\Mateusz\Desktop\dataset_bad\11.jpg')
#
# r, g, b = im.split()
# r= r.crop((92,92,r.width-92,r.height-92))
#
# im2 = Image.open(r'C:\Users\Mateusz\Desktop\dataset_100\11.jpg')
# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
#
# r2, g, b = im2.split()
# x = np.array(ImageChops.difference(im, im2))
# print(x)
# x = (x<7) * 1
# x = rgb2gray(x)
# print(x)
# plt.imshow(x, cmap='gray', vmin=0, vmax=1)
# plt.show()
# plt.imshow(x, cmap='gray', vmin=0, vmax=1)
# plt.show()

# with tf.device('/CPU:0'):
batch_size = 3
image_size = 256

myGene = trainGenerator(size=image_size, batch_size=batch_size)

model = unet(shape=(image_size, image_size, 1), f=image_size/2, k=image_size/16, batch_size=batch_size)

# model:Model = tf.keras.models.load_model(r'C:\Users\Mateusz\Desktop')
# img = Image.open(r"C:\Users\Mateusz\Desktop\dataset_bad\11.jpg")
# img = np.array(img)
# img = np.array([img, img, img])
# img = np.expand_dims(img, axis=0)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=r"C:\Users\Mateusz\Desktop\{epoch:02d}-{val_accuracy:.2f}.hdf5",
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)
model.fit(myGene, batch_size=batch_size, epochs=100, steps_per_epoch=244, callbacks=[model_checkpoint_callback])
# y = model.predict(img, batch_size=1)
# print(img)
# print(y.shape)
# print(y[0, :, :, 0])
# plt.imshow(y[0])
# plt.show()
# x = y[0, :, :, 0]-img[0]
# print(x)
# plt.imshow(x, cmap='gray')
# plt.show()
model.save(r'C:\Users\Mateusz\Desktop\2')
