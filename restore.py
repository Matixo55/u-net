import numpy as np
from PIL import Image

from model import unet

batch_size = 3
image_size = 256
neurons_number = 64


def process_overlaping_areas(array, new, y, x, size, overlap_size, x_max=None, y_max=None):
    step = 1 / (overlap_size + 1)
    # Przetwarzanie do końca segmentu lub do końca obrazu
    x_max = x_max or x + size
    y_max = y_max or y + size
    coef_prev_segment = 1 - step
    coef_new_segment = step

    for i in range(overlap_size):
        # Oblicz wartość pixeli wyjściowych w pokrywających się rzędach
        if y != 0:
            array[y + i, x : x + size] = (
                coef_prev_segment * array[y + i, x : x + size] + coef_new_segment * new[i, :x_max]
            )
        else:
            # Dla pierwszego rzędu nie ma wyższego rzędu do uwzględnienia
            array[y + i, x : x + size] = new[i, :x_max]
        coef_prev_segment -= step
        coef_new_segment += step

    coef_prev_segment = 1 - step
    coef_new_segment = step

    for i in range(overlap_size):
        # Oblicz wartość pixeli wyjściowych w pokrywających się kolumnach
        if x != 0:
            array[y : y + size, x + i] = (
                coef_prev_segment * array[y : y + size, x + i] + coef_new_segment * new[:y_max, i]
            )
        else:
            # Dla pierwszej kolumny nie ma poprzedniej kolumny do uwzględnienia
            array[y : y + size, x + i] = new[:y_max, i]
        coef_prev_segment -= step
        coef_new_segment += step

    return array


def restore_segment(array, model):
    new = np.array([array, array, array])
    # Model przystosowany jest do paczek o wielkości 3
    new = np.expand_dims(new, axis=3)
    new = model.predict(new)[0, :, :, 0]
    # Zaokrąglenie do liczb całkowitych
    new = np.rint(new)
    # Winsoryazcja do zakresu 0-255
    new[new > 255] = 255
    new[new < 0] = 0
    return new.astype(np.uint8)


def restore_middle_segment(array, x, y, size, model, overlap_size):
    new = array[y : y + size, x : x + size]
    new = restore_segment(new, model)

    array[y + overlap_size : y + size, x + overlap_size : x + size] = new[
        overlap_size:, overlap_size:
    ]
    return process_overlaping_areas(array, new, y, x, size, overlap_size)


def restore_side_segment(array, x, y, size, image, model, overlap_size):
    x_max = image.size[0]
    missing_x = size - (x_max - x)

    if missing_x == 0:
        return restore_middle_segment(array, x, y, size, model, overlap_size)

    new = np.zeros((size, size))
    new[:, : x_max - x] = array[y : y + size, x:]
    # Uzupełnienie brakujących kolumn lustrzanym odbiciem obrazu
    flipped_x = np.fliplr(array[y : y + size, -missing_x:])
    new[:, -missing_x:] = flipped_x

    new = restore_segment(new, model)
    array[y + overlap_size : y + size, x + overlap_size :] = new[
        overlap_size:, overlap_size : x_max - x
    ]
    return process_overlaping_areas(array, new, y, x, size, overlap_size, x_max=x_max - x)


def restore_bottom_segment(array, x, y, size, image, model, overlap_size):
    y_max = image.size[1]
    missing_y = size - (y_max - y)

    if missing_y == 0:
        return restore_middle_segment(array, x, y, size, model, overlap_size)

    new = np.zeros((size, size))
    new[: y_max - y, :] = array[y:, x : x + size]
    # Uzupełnienie brakujących wierszy lustrzanym odbiciem obrazu
    flipped_x = np.fliplr(array[-missing_y:, x : x + size])
    new[-missing_y:, :] = flipped_x

    new = restore_segment(new, model)

    array[y + overlap_size :, x + overlap_size : x + size] = new[
        overlap_size : y_max - y, overlap_size:
    ]
    return process_overlaping_areas(array, new, y, x, size, overlap_size, y_max=y_max - y)


def restore_corner_segment(array, x, y, size, image, model, overlap_size):
    x_max = image.size[0]
    y_max = image.size[1]
    missing_x = size - (x_max - x)
    missing_y = size - (y_max - y)

    if missing_x == 0:
        return restore_bottom_segment(array, x, y, size, image, model, overlap_size)
    if missing_y == 0:
        return restore_side_segment(array, x, y, size, image, model, overlap_size)

    new = np.zeros((size, size))
    new[: y_max - y, : x_max - x] = array[y:, x:]

    flipped_x = np.fliplr(array[-missing_y:, -missing_x:])
    # Uzupełnienie brakujących kolumn i wierszy lustrzanym odbiciem obrazu
    new[-missing_y:, -missing_x:] = flipped_x
    new = restore_segment(new, model)

    array[y + overlap_size :, x + overlap_size :] = new[
        overlap_size : y_max - y, overlap_size : x_max - x
    ]
    return process_overlaping_areas(
        array, new, y, x, size, overlap_size, x_max=x_max - x, y_max=y_max - y
    )


def restore_image(image, size, model, original, overlap_size):
    red, green, blue = image.split()
    red = np.array(red)
    green = np.array(green)
    blue = np.array(blue)
    # r, g, b = original.split()
    # r = np.array(r)
    # g = np.array(g)
    # b = np.array(b)

    x = 0
    y = 0
    while y + size < image.size[1]:
        print(y)
        while x + size < image.size[0]:
            red = restore_middle_segment(red, x, y, size, model, overlap_size)
            blue = restore_middle_segment(blue, x, y, size, model, overlap_size)
            green = restore_middle_segment(green, x, y, size, model, overlap_size)

            x += size - overlap_size

        if x != image.size[0]:
            # Jeżeli szerokość obrazu nie jest wielokrotnością rozmiaru segmentu, pozostanie fragment na brzegu
            red = restore_side_segment(red, x, y, size, image, model, overlap_size)
            blue = restore_side_segment(blue, x, y, size, image, model, overlap_size)
            green = restore_side_segment(green, x, y, size, image, model, overlap_size)

        x = 0
        y += size - overlap_size

    if y != image.size[1]:
        # Jeżeli wysokość obrazu nie jest wielokrotnością rozmiaru segmentu, pozostanie fragment na dole
        print(y)
        while x + size < image.size[0]:
            red = restore_bottom_segment(red, x, y, size, image, model, overlap_size)
            blue = restore_bottom_segment(blue, x, y, size, image, model, overlap_size)
            green = restore_bottom_segment(green, x, y, size, image, model, overlap_size)

            x += size - overlap_size

        if x != image.size[0]:
            # Dodatkowo może pozostać fragment w rogu obrazu
            red = restore_corner_segment(red, x, y, size, image, model, overlap_size)
            blue = restore_corner_segment(blue, x, y, size, image, model, overlap_size)
            green = restore_corner_segment(green, x, y, size, image, model, overlap_size)

    image = np.stack([red, green, blue], axis=2)
    return image


def run(overlap_size, filename):
    model = unet(
        shape=(image_size, image_size, 1),
        neurons_number=neurons_number,
        batch_size=batch_size,
        pretrained_weights=r".\weights\92-epoch-1.9141-loss-5-layers.hdf5",
    )

    img = Image.open(rf".\original_compressed\{filename}.jpg")
    original = Image.open(rf".\original\{filename}.jpg")
    img = restore_image(img, image_size, model, original, overlap_size)
    img = Image.fromarray(img)  # .convert("L")
    img.save(rf".\samples\{filename}.jpg", quality=100)


run(overlap_size=8, filename="01")
