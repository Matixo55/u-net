import tensorflow as tf

from data import *
from model import unet

batch_size = 3
image_size = 256
n_samples = 356
f = 64  # image_size/2
k = 8  # image_size/16

generator = training_data_generator(size=image_size, batch_size=batch_size)

model = unet(
    shape=(image_size, image_size, 1),
    f=f,
    k=k,
    batch_size=batch_size,
    # pretrained_weights=r".\weights\44-epoch-5-layers-128-0.036loss.hdf5",
)


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=r".\weights\{epoch:02d}-epoch-{loss:.4f}-loss-{accuracy:.2f}-accuracy.hdf5",
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)
model.fit(
    generator,
    batch_size=batch_size,
    steps_per_epoch=int(n_samples / (batch_size)),
    epochs=1000,
    callbacks=[model_checkpoint_callback],
)
model.save("model")
