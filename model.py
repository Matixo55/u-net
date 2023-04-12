from PIL.Image import Image
from keras.utils import plot_model, model_to_dot
from matplotlib import pyplot as plt
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.optimizer_v2.adam import Adam
from visualkeras import layered_view


def encoder_layer(input, neurons_number):
    pooling = layers.MaxPooling2D(pool_size=2, strides=2)(input)
    conv = layers.Conv2D(neurons_number, kernel_size=3, padding="same", activation="relu")(pooling)
    return layers.Conv2D(neurons_number, kernel_size=3, padding="same", activation="relu")(conv)


def decoder_layer(input, neurons_number, connected_layer):
    tconv = layers.Conv2DTranspose(
        neurons_number, kernel_size=2, strides=(2, 2), activation="relu"
    )(input)
    concat = layers.Concatenate(axis=3)([tconv, connected_layer])
    dconv = layers.Conv2D(neurons_number / 2, kernel_size=3, padding="same", activation="relu")(
        concat
    )
    return layers.Conv2D(neurons_number / 2, kernel_size=3, padding="same", activation="relu")(
        dconv
    )


def unet(shape, neurons_number: int, batch_size: int, pretrained_weights=None):
    # -------------------------------------------------------------------------

    input_layer = layers.Input(shape=shape, batch_size=batch_size)
    conv = layers.Conv2D(neurons_number, kernel_size=3, padding="same", activation="relu")(
        input_layer
    )
    encoder_0 = layers.Conv2D(neurons_number, kernel_size=3, padding="same", activation="relu")(
        conv
    )

    # -------------------------------------------------------------------------
    neurons_number *= 2

    encoder_1 = encoder_layer(encoder_0, neurons_number)

    # -------------------------------------------------------------------------
    neurons_number *= 2

    encoder_2 = encoder_layer(encoder_1, neurons_number)

    # -------------------------------------------------------------------------
    neurons_number *= 2

    encoder_3 = encoder_layer(encoder_2, neurons_number)

    # -------------------------------------------------------------------------
    # neurons_number *= 2

    # encoder_4 = encoder_layer(encoder_3, neurons_number)

    # -------------------------------------------------------------------------
    # neurons_number /= 2
    #
    # decoder_1 = decoder_layer(encoder_3, neurons_number, encoder_3)

    # -------------------------------------------------------------------------
    neurons_number /= 2

    decoder_2 = decoder_layer(encoder_3, neurons_number, encoder_2)

    # -------------------------------------------------------------------------
    neurons_number /= 2

    decoder_3 = decoder_layer(decoder_2, neurons_number, encoder_1)

    # -------------------------------------------------------------------------
    neurons_number /= 2

    decoder_4 = decoder_layer(decoder_3, neurons_number, encoder_0)

    # -------------------------------------------------------------------------

    flatten = layers.Conv2D(1, kernel_size=1)(decoder_4)
    output = layers.Add()([flatten, input_layer])

    # -------------------------------------------------------------------------

    model = models.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="mean_absolute_error",
        metrics=[],
        run_eagerly=True,
    )
    # layered_view(model, to_file="model_plot.png",legend=True, draw_volume=False, max_xy=8000)
    model_to_dot(model, to_file="model_plot.png", show_shapes=True, show_layer_names=False, dpi=200, show_layer_activations=True)

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
