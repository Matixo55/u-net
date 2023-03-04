from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator


def trainGenerator(size: int, batch_size: int):
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        r"C:\Users\Mateusz\Desktop",
        classes=["dataset_bad"],
        target_size=(size, size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode=None,
        seed=0,
    )
    mask_generator = mask_datagen.flow_from_directory(
        r"C:\Users\Mateusz\Desktop",
        classes=["dataset_100"],
        target_size=(size, size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode=None,
        seed=0,
    )
    train_generator = zip(image_generator, mask_generator)
    for img, mask in train_generator:
        yield img, mask
