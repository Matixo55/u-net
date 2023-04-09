from tensorflow.python.keras import layers, models
from tensorflow.python.keras.optimizer_v2.adam import Adam


def conv_layer(input_x, f):
    flat_1 = layers.Conv2D(f, kernel_size=3, padding="same", activation="relu")(input_x)
    return layers.Conv2D(f, kernel_size=3, padding="same", activation="relu")(flat_1)


def dconv_layer(input, f, old_deep_layer):
    dconv_1 = layers.Conv2DTranspose(f, kernel_size=2, strides=(2, 2))(input)
    dconv_1 = layers.Activation("relu")(dconv_1)
    dconv_1 = layers.Concatenate(axis=3)([dconv_1, old_deep_layer])

    dconv_flat = layers.Conv2D(f / 2, kernel_size=3, padding="same", activation="relu")(
        dconv_1
    )
    return layers.Conv2D(f / 2, kernel_size=3, padding="same", activation="relu")(
        dconv_flat
    )


def unet(shape, f: int, k: int, batch_size: int, pretrained_weights=None):
    x_input = layers.Input(shape=shape, batch_size=batch_size)

    # -------------------------------------------------------------------------

    l = 1

    deep_1 = conv_layer(x_input, f)

    # -------------------------------------------------------------------------

    f *= 2
    l = 2

    conv_2 = layers.MaxPooling2D(pool_size=2, strides=2)(deep_1)
    deep_2 = conv_layer(conv_2, f)

    # -------------------------------------------------------------------------

    f *= 2
    l = 3

    conv_3 = layers.MaxPooling2D(pool_size=2, strides=2)(deep_2)
    deep_3 = conv_layer(conv_3, f)

    # -------------------------------------------------------------------------
    #
    f *= 2
    l = 4

    conv_4 = layers.MaxPooling2D(pool_size=2, strides=2)(deep_3)
    deep_4 = conv_layer(conv_4, f)

    # -------------------------------------------------------------------------

    f *= 2
    l = 5

    conv_5 = layers.MaxPooling2D(pool_size=2, strides=2)(deep_4)
    deep_5 = conv_layer(conv_5, f)

    # -------------------------------------------------------------------------

    f /= 2
    l = 4

    dconv_1 = dconv_layer(deep_5, f, deep_4)

    # -------------------------------------------------------------------------

    f /= 2
    l = 3

    dconv_2 = dconv_layer(dconv_1, f, deep_3)

    # -------------------------------------------------------------------------

    f /= 2
    l = 2

    dconv_3 = dconv_layer(dconv_2, f, deep_2)

    # -------------------------------------------------------------------------

    f /= 2
    l = 1

    dconv_4 = dconv_layer(dconv_3, f, deep_1)

    # -------------------------------------------------------------------------

    x_output = layers.Conv2D(1, kernel_size=1)(dconv_4)
    output = layers.Add()([x_output, x_input])

    model = models.Model(inputs=x_input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="mean_absolute_error",
        metrics=[],
        run_eagerly=True,
    )

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
