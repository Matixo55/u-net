import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.python.keras.saving.save import load_model

from model import unet
import numpy as np


batch_size = 3
image_size = 256
f = 64
k = 8


def predict_array(array, model):
    new = np.array([array, array, array])
    new = np.expand_dims(new, axis=3)
    new = model.predict(new)[0, :, :, 0]
    new = np.rint(new)
    new[new > 255] = 255
    new[new < 0] = 0
    new = new.astype(np.uint8)
    return new


def replace_middle_array(array, x, y, size, model):
    new = array[y:y + size, x:x + size]
    new = predict_array(new, model)
    array[y:y + size, x:x + size] = new
    return array


def replace_side_array(array, x, y, size, image, model):
    x_max = image.size[0]
    new = np.zeros((size, size))
    missing_x = size-(x_max - x)

    new[:, :x_max - x] = array[y:y + size, x:]
    flipped_x = np.fliplr(array[y:y + size, -missing_x:])
    new[:, -missing_x:] = flipped_x
    new = predict_array(new, model)
    array[y:y + size, x:] = new[:, :x_max - x]
    return array


def replace_bottom_array(array, x, y, size, image, model):
    y_max = image.size[1]
    new = np.zeros((size, size))
    missing_y = size-(y_max - y)

    new[:y_max - y, :] = array[y:, x:x+size]
    flipped_x = np.fliplr(array[-missing_y:, x:x+size])
    new[-missing_y:, :] = flipped_x
    new = predict_array(new, model)
    array[y:, x:x+size] = new[:y_max - y, :]
    return array


def replace_corner_array(array, x, y, size, image, model):
    x_max = image.size[0]
    y_max = image.size[1]
    new = np.zeros((size, size))
    missing_x = size-(x_max - x)
    missing_y = size-(y_max - y)

    new[:y_max - y, :x_max - x] = array[y:, x:]
    flipped_x = np.fliplr(array[-missing_y:, -missing_x:])
    new[-missing_y:, -missing_x:] = flipped_x
    new = predict_array(new, model)
    array[y:, x:] = new[:y_max - y, :x_max - x]
    return array


def restore(image, size, model):
    o_r, o_g, o_b = image.split()
    o_r = np.array(o_r)
    o_g = np.array(o_g)
    o_b = np.array(o_b)
    x = 0
    y = 0
    while y + size < image.size[1]:
        print(y)
        while x + size < image.size[0]:
            o_r = replace_middle_array(o_r, x, y, size, model)
            o_b = replace_middle_array(o_b, x, y, size, model)
            o_g = replace_middle_array(o_g, x, y, size, model)

            x += size

        if x != image.size[0]:
            o_r = replace_side_array(o_r, x, y, size, image, model)
            o_b = replace_side_array(o_b, x, y, size, image, model)
            o_g = replace_side_array(o_g, x, y, size, image, model)

        x = 0
        y += size

    if y != image.size[1]:
        print(y)
        while x + size < image.size[0]:
            o_r = replace_bottom_array(o_r, x, y, size, image, model)
            o_b = replace_bottom_array(o_b, x, y, size, image, model)
            o_g = replace_bottom_array(o_g, x, y, size, image, model)

            x += size

        if x != image.size[0]:
            o_r = replace_corner_array(o_r, x, y, size, image, model)
            o_b = replace_corner_array(o_b, x, y, size, image, model)
            o_g = replace_corner_array(o_g, x, y, size, image, model)

    image = np.stack([o_r, o_g, o_b], axis=2)
    plt.imshow(image)
    plt.show()
    return image


model = unet(
    shape=(image_size, image_size, 1),
    f=f,
    k=k,
    batch_size=batch_size,
    pretrained_weights=r".\weights\24-epoch-2.4550-loss-256-real.hdf5",
)

# model = tf.keras.models.load_model(r'model')
img = Image.open(r"samples\4k_compressed.jpg")
img = restore(img, image_size, model)
img = Image.fromarray(img)#.convert("L")
img.save(r"samples\4k_restored.jpg", quality=100)
exit()
r, g, b = img.split()
# img = np.array(img)[:, :, 0]
# print(img.shape)
# r = Image.open(r".\compressed\73.jpg")
# g = Image.open(r".\compressed\96.jpg")
# b = Image.open(r".\compressed\119.jpg")
r = np.array(r)
g = np.array(g)
b = np.array(b)
r = np.array([r, r, r])
r = np.expand_dims(r, axis=3)
g = np.array([g, g, g])
g = np.expand_dims(g, axis=3)
b = np.array([b, b, b])
b = np.expand_dims(b, axis=3)
# img = np.array([img, img, img])
# img = np.expand_dims(img, axis=3)
r = model.predict(r)[0, :, :, 0]
g = model.predict(g)[0, :, :, 0]
b = model.predict(b)[0, :, :, 0]

r = np.rint(r)
r[r> 255] = 255
r[r< 0] = 0
r = r.astype(np.uint8)

g = np.rint(g)
g[g> 255] = 255
g[g< 0] = 0
g = g.astype(np.uint8)

b = np.rint(b)
b[b> 255] = 255
b[b< 0] = 0
b = b.astype(np.uint8)

img = np.stack([r, g, b], axis=2)
print(img.shape)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.show()

img = Image.fromarray(img)#.convert("L")
img.save(r".\samples\test.jpg", quality=100)
