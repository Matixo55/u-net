import matplotlib.pyplot as plt
import tensorflow as tf

from data import *
from model import unet
from dense_model import dense_unet

batch_size = 3
image_size = 256
n_samples = 9515
n_validation_samples = 997
epochs = 100
f = 64  # image_size/2
k = 8  # image_size/16
filename = "flat-2"

train_data_generator = data_generator(
    size=image_size, batch_size=batch_size, type="train"
)
validation_data_generator = data_generator(
    size=image_size, batch_size=batch_size, type="validation"
)

model = unet(
    shape=(image_size, image_size, 1),
    f=f,
    k=k,
    batch_size=batch_size,
    # pretrained_weights=r".\weights\24-epoch-2.4550-loss-256-real.hdf5",
)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=r".\weights\{epoch:02d}-epoch-{loss:.4f}-loss.hdf5",
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

history = model.fit(
    train_data_generator,
    batch_size=batch_size,
    steps_per_epoch=int(n_samples / (batch_size * 10)),
    epochs=epochs,
    callbacks=[model_checkpoint_callback],
    validation_steps=int(n_validation_samples / batch_size),
    validation_data=validation_data_generator,
    validation_batch_size=batch_size,
)

plt.plot(history.history["loss"], color="orange", label="train")
plt.plot(history.history["val_loss"], color="blue", label="validation")
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.savefig(f"{filename}.png")

model.save("model")

with open(f"{filename}.txt", "w") as f:
    f.write(str(history.history["loss"]) + "\n")
    f.write(str(history.history["val_loss"]))
