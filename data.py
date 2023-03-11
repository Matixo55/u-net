from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator


def training_data_generator(size: int, batch_size: int):
    image_generator = ImageDataGenerator().flow_from_directory(
        ".",
        classes=["compressed"],
        target_size=(size, size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode=None,
        seed=0,
    )
    mask_generator = ImageDataGenerator().flow_from_directory(
        ".",
        classes=["generated"],
        target_size=(size, size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode=None,
        seed=0,
    )

    train_generator = zip(image_generator, mask_generator)

    for img, mask in train_generator:
        yield img, mask
