import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.python.keras.saving.save import load_model

from model import unet
from data import *
import numpy as np


batch_size = 3
image_size = 256
n_samples = 356
f = 64
k = 8

generator = training_data_generator(size=image_size, batch_size=batch_size)

model = unet(
    shape=(image_size, image_size, 1),
    f=f,
    k=k,
    batch_size=batch_size,
    pretrained_weights=r".\weights\44-epoch-5-layers-128-0.036loss.hdf5",
)

# model = tf.keras.models.load_model(r'model')
img = Image.open(r".\samples\compressed_simple.jpg")
plt.imshow(img, cmap="gray")
plt.show()
img = np.array(img)[:, :, 0]
# r, g, b = img.split()
# r = np.array(r)
# g = np.array(g)
# b = np.array(b)
# r = np.array([r, r, r])
# r = np.expand_dims(r, axis=3)
# g = np.array([g, g, g])
# g = np.expand_dims(g, axis=3)
# b = np.array([b, b, b])
# b = np.expand_dims(b, axis=3)
img = np.array([img, img, img])
img = np.expand_dims(img, axis=3)
img = model.predict(img)[0, :, :, 0]
img = np.rint(img).astype(int)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.show()

# img = Image.fromarray(img).convert("RGB")
# img.save(r".\samples\restored_simple.jpg")
