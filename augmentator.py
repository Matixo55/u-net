import os

from PIL import Image

original = r".\original"
generated = r".\generated"
compressed = r".\compressed"
pk = 0
size = 256


def generate(image, size, pk):
    x = 0
    y = 0
    while y + size < image.size[1]:
        while x + size < image.size[0]:
            cropped_image = image.crop((x, y, x + size, y + size))
            cropped_image.save(os.path.join(generated, str(pk) + ".jpg"), quality=100)
            cropped_image.save(os.path.join(compressed, str(pk) + ".jpg"), quality=25)
            pk += 1
            x += size
        y += size

    return pk


for filename in os.listdir(original):
    file = os.path.join(original, filename)

    if not os.path.isfile(file):
        continue

    print(filename)

    img = Image.open(file)

    rotations = [img, img.transpose(1), img.transpose(0), img.transpose(1).transpose(0)]

    for image in rotations:
        r, g, b = image.split()
        pk = generate(r, size, pk)
        pk = generate(g, size, pk)
        pk = generate(b, size, pk)
