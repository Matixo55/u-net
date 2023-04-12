import matplotlib.pyplot as plt
import tensorflow as tf

from data import data_generator
from model import unet

batch_size = 3
sample_size = 256
test_samples_number = 9515
validation_samples_number = 997
epochs = 100
neurons_number = 64
steps_divider = 10
filename = "5-layers"

train_data_generator = data_generator(size=sample_size, batch_size=batch_size, type="train")
validation_data_generator = data_generator(
    size=sample_size, batch_size=batch_size, type="validation"
)

model = unet(
    shape=(sample_size, sample_size, 1),
    neurons_number=neurons_number,
    batch_size=batch_size,
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
    steps_per_epoch=int(test_samples_number / (batch_size * steps_divider)),
    epochs=epochs,
    callbacks=[model_checkpoint_callback],
    validation_steps=int(validation_samples_number / batch_size),
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

with open(f"{filename}.txt", "w") as neurons:
    neurons.write(str(history.history["loss"]) + "\n")
    neurons.write(str(history.history["val_loss"]))
