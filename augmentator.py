import os
import random

from PIL import Image

original = r".\original"
generated = r".\generated"
compressed = r".\compressed"
pk = 0
size = 256
validation_pk = 0


def generate(image, size, pk, folder, validation_pk):
    x = 0
    y = 0
    while y + size < image.size[1]:
        while x + size < image.size[0]:
            cropped_image = image.crop((x, y, x + size, y + size))

            if folder == "train" and random.randint(0, 10) == 5:
                validation_pk += 1
                cropped_image.save(
                    os.path.join(generated, "validation", str(pk) + ".jpg"), quality=100
                )
                cropped_image.save(
                    os.path.join(compressed, "validation", str(pk) + ".jpg"), quality=25
                )
            else:
                cropped_image.save(
                    os.path.join(generated, folder, str(pk) + ".jpg"), quality=100
                )
                cropped_image.save(
                    os.path.join(compressed, folder, str(pk) + ".jpg"), quality=25
                )
                pk += 1
            x += size
        y += size

    return pk, validation_pk


for filename in os.listdir(os.path.join(generated, "test")):
    os.remove(os.path.join(generated, "test", filename))

for filename in os.listdir(os.path.join(generated, "train")):
    os.remove(os.path.join(generated, "train", filename))

for filename in os.listdir(os.path.join(generated, "validation")):
    os.remove(os.path.join(generated, "validation", filename))

for filename in os.listdir(os.path.join(compressed, "test")):
    os.remove(os.path.join(compressed, "test", filename))

for filename in os.listdir(os.path.join(compressed, "train")):
    os.remove(os.path.join(compressed, "train", filename))

for filename in os.listdir(os.path.join(compressed, "validation")):
    os.remove(os.path.join(compressed, "validation", filename))


for filename in os.listdir(original):
    file = os.path.join(original, filename)

    if not os.path.isfile(file) or filename[0] == "t":
        continue

    print(filename)

    img = Image.open(file)

    rotations = [
        img,
        img.transpose(1),
        img.transpose(0),
        img.transpose(1).transpose(0),
        img.transpose(2),
        img.transpose(2).transpose(1),
        img.transpose(2).transpose(0),
        img.transpose(2).transpose(1).transpose(0),
    ]

    for image in rotations:
        r, g, b = image.split()
        pk, validation_pk = generate(r, size, pk, "train", validation_pk)
        pk, validation_pk = generate(g, size, pk, "train", validation_pk)
        pk, validation_pk = generate(b, size, pk, "train", validation_pk)


print(pk)
print(validation_pk)


for filename in os.listdir(original):
    file = os.path.join(original, filename)

    if not os.path.isfile(file) or filename[0] != "t":
        continue

    print(filename)

    img = Image.open(file)

    rotations = [
        img,
        img.transpose(1),
        img.transpose(0),
        img.transpose(1).transpose(0),
        img.transpose(2),
        img.transpose(2).transpose(1),
        img.transpose(2).transpose(0),
        img.transpose(2).transpose(1).transpose(0),
    ]

    for image in rotations:
        r, g, b = image.split()
        generate(r, size, pk, "test", validation_pk)
        generate(g, size, pk, "test", validation_pk)
        generate(b, size, pk, "test", validation_pk)
