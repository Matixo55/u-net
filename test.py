import numpy as np
from PIL import Image

from model import unet

batch_size = 3
image_size = 256
f = 64
k = 8


def gradient(array, new, y, x, size, gradient_size, x_max=None, y_max=None):
    step = 1 / (gradient_size + 1)
    x_max = x_max or x + size
    y_max = y_max or y + size
    coef_old = 1 - step
    coef_new = step

    for i in range(gradient_size):
        if y != 0:
            array[y + i, x : x + size] = (
                coef_old * array[y + i, x : x + size] + coef_new * new[i, :x_max]
            )
        else:
            array[y + i, x : x + size] = new[i, :x_max]
        coef_old -= step
        coef_new += step

    coef_old = 1 - step
    coef_new = step

    for i in range(gradient_size):
        if x != 0:
            array[y : y + size, x + i] = (
                coef_old * array[y : y + size, x + i] + coef_new * new[:y_max, i]
            )
        else:
            array[y : y + size, x + i] = new[:y_max, i]
        coef_old -= step
        coef_new += step

    return array


def predict_array(array, model):
    new = np.array([array, array, array])
    new = np.expand_dims(new, axis=3)
    new = model.predict(new)[0, :, :, 0]
    new = np.rint(new)
    new[new > 255] = 255
    new[new < 0] = 0
    new = new.astype(np.uint8)
    return new


def replace_middle_array(array, x, y, size, model, gradient_size):
    new = array[y : y + size, x : x + size]
    new = predict_array(new, model)

    array[y + gradient_size : y + size, x + gradient_size : x + size] = new[
        gradient_size:, gradient_size:
    ]
    array = gradient(array, new, y, x, size, gradient_size)

    return array


def replace_side_array(array, x, y, size, image, model, gradient_size):
    x_max = image.size[0]
    missing_x = size - (x_max - x)

    if missing_x == 0:
        return replace_middle_array(array, x, y, size, model, gradient_size)

    new = np.zeros((size, size))
    new[:, : x_max - x] = array[y : y + size, x:]
    flipped_x = np.fliplr(array[y : y + size, -missing_x:])
    new[:, -missing_x:] = flipped_x
    new = predict_array(new, model)

    array[y + gradient_size : y + size, x + gradient_size :] = new[
        gradient_size:, gradient_size : x_max - x
    ]
    array = gradient(array, new, y, x, size, gradient_size, x_max=x_max - x)

    return array


def replace_bottom_array(array, x, y, size, image, model, gradient_size):
    y_max = image.size[1]
    missing_y = size - (y_max - y)

    if missing_y == 0:
        return replace_middle_array(array, x, y, size, model, gradient_size)

    new = np.zeros((size, size))
    new[: y_max - y, :] = array[y:, x : x + size]
    flipped_x = np.fliplr(array[-missing_y:, x : x + size])
    new[-missing_y:, :] = flipped_x
    new = predict_array(new, model)

    array[y + gradient_size :, x + gradient_size : x + size] = new[
        gradient_size : y_max - y, gradient_size:
    ]
    array = gradient(array, new, y, x, size, gradient_size, y_max=y_max - y)

    return array


def replace_corner_array(array, x, y, size, image, model, gradient_size):
    x_max = image.size[0]
    y_max = image.size[1]
    missing_x = size - (x_max - x)
    missing_y = size - (y_max - y)

    if missing_x == 0:
        return replace_bottom_array(array, x, y, size, image, model, gradient_size)
    if missing_y == 0:
        return replace_side_array(array, x, y, size, image, model, gradient_size)

    new = np.zeros((size, size))
    new[: y_max - y, : x_max - x] = array[y:, x:]

    flipped_x = np.fliplr(array[-missing_y:, -missing_x:])
    new[-missing_y:, -missing_x:] = flipped_x
    new = predict_array(new, model)

    array[y + gradient_size :, x + gradient_size :] = new[
        gradient_size : y_max - y, gradient_size : x_max - x
    ]
    array = gradient(
        array, new, y, x, size, gradient_size, x_max=x_max - x, y_max=y_max - y
    )

    return array


def restore(image, size, model, original, gradient_size):
    o_r, o_g, o_b = image.split()
    o_r = np.array(o_r)
    o_g = np.array(o_g)
    o_b = np.array(o_b)
    # a, b, c = original.split()
    # a = np.array(a)
    # b = np.array(b)
    # c = np.array(c)

    x = 0
    y = 0
    while y + size < image.size[1]:
        print(y)
        while x + size < image.size[0]:
            o_r = replace_middle_array(o_r, x, y, size, model, gradient_size)
            o_b = replace_middle_array(o_b, x, y, size, model, gradient_size)
            o_g = replace_middle_array(o_g, x, y, size, model, gradient_size)

            x += size - gradient_size

        if x != image.size[0]:
            o_r = replace_side_array(o_r, x, y, size, image, model, gradient_size)
            o_b = replace_side_array(o_b, x, y, size, image, model, gradient_size)
            o_g = replace_side_array(o_g, x, y, size, image, model, gradient_size)

        x = 0
        y += size - gradient_size

    if False and y != image.size[1]:
        print(y)
        while x + size < image.size[0]:
            o_r = replace_bottom_array(o_r, x, y, size, image, model, gradient_size)
            o_b = replace_bottom_array(o_b, x, y, size, image, model, gradient_size)
            o_g = replace_bottom_array(o_g, x, y, size, image, model, gradient_size)

            x += size - gradient_size

        if x != image.size[0]:
            o_r = replace_corner_array(o_r, x, y, size, image, model, gradient_size)
            o_b = replace_corner_array(o_b, x, y, size, image, model, gradient_size)
            o_g = replace_corner_array(o_g, x, y, size, image, model, gradient_size)
    # print(
    #     f"Gradient {gradient_size}  {mean(mean(np.abs(a - o_r), axis=-1)+ mean(np.abs(b - o_g), axis=-1)+ mean(np.abs(c - o_b), axis=-1)):.2f}"
    # )

    image = np.stack([o_r, o_g, o_b], axis=2)
    return image


def run(gradient_size, filename):
    model = unet(
        shape=(image_size, image_size, 1),
        f=f,
        k=k,
        batch_size=batch_size,
        pretrained_weights=r".\weights\92-epoch-1.9141-loss-5-layers.hdf5",
    )

    # model = tf.keras.models.load_model(r'model')
    img = Image.open(rf".\original_compressed\{filename}.jpg")
    original = Image.open(rf".\original\{filename}.jpg")
    img = restore(img, image_size, model, original, gradient_size)
    img = Image.fromarray(img)  # .convert("L")
    img.save(rf".\samples\{filename}.jpg", quality=100)


run(gradient_size=8, filename=r"calibration_chart")
