from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator


def data_generator(size: int, batch_size: int, type: str):
    image_generator = ImageDataGenerator().flow_from_directory(
        r"./compressed",
        classes=[type],
        target_size=(size, size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode=None,
        seed=0,
    )
    mask_generator = ImageDataGenerator().flow_from_directory(
        r"./generated",
        classes=[type],
        target_size=(size, size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode=None,
        seed=0,
    )

    for img, mask in zip(image_generator, mask_generator):
        yield img, mask
