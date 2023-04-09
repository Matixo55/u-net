from tensorflow.python.keras import layers, models
from tensorflow.python.keras.optimizer_v2.adam import Adam


def deep_layer(input_x, f, k):
    F = f / 2

    deep_1a = layers.Conv2D(F, kernel_size=3, padding="same", use_bias=False)(input_x)
    deep_1 = layers.Conv2D(F, kernel_size=1, padding="same", use_bias=False)(deep_1a)
    deep_1 = layers.BatchNormalization()(deep_1)
    deep_1 = layers.Activation("relu")(deep_1)
    deep_1 = layers.Conv2D(k, kernel_size=3, padding="same", use_bias=False)(deep_1)
    deep_1 = layers.BatchNormalization()(deep_1)
    deep_1 = layers.Activation("relu")(deep_1)

    deep_2 = layers.Concatenate(axis=3)([deep_1a, deep_1])
    deep_2 = layers.Conv2D(F, kernel_size=1, padding="same", use_bias=False)(deep_2)
    deep_2 = layers.BatchNormalization()(deep_2)
    deep_2 = layers.Activation("relu")(deep_2)
    deep_2 = layers.Conv2D(k, kernel_size=3, padding="same", use_bias=False)(deep_2)
    deep_2 = layers.BatchNormalization()(deep_2)
    deep_2 = layers.Activation("relu")(deep_2)

    deep_3 = layers.Concatenate(axis=3)([deep_1a, deep_1, deep_2])
    deep_3 = layers.Conv2D(F, kernel_size=1, padding="same", use_bias=False)(deep_3)
    deep_3 = layers.BatchNormalization()(deep_3)
    deep_3 = layers.Activation("relu")(deep_3)
    deep_3 = layers.Conv2D(k, kernel_size=3, padding="same", use_bias=False)(deep_3)
    deep_3 = layers.BatchNormalization()(deep_3)
    deep_3 = layers.Activation("relu")(deep_3)

    deep_4 = layers.Concatenate(axis=3)([deep_1a, deep_1, deep_2, deep_3])
    deep_4 = layers.Conv2D(F, kernel_size=1, padding="same", use_bias=False)(deep_4)
    deep_4 = layers.BatchNormalization()(deep_4)
    deep_4 = layers.Activation("relu")(deep_4)
    deep_4 = layers.Conv2D(k, kernel_size=3, padding="same", use_bias=False)(deep_4)
    deep_4 = layers.BatchNormalization()(deep_4)
    deep_4 = layers.Activation("relu")(deep_4)

    return layers.Concatenate(axis=3)([deep_1a, deep_1, deep_2, deep_3, deep_4])


def dconv_layer(input, f, k, old_deep_layer):
    dconv_1 = layers.Conv2DTranspose(f, kernel_size=2, strides=(2, 2))(input)
    dconv_1 = layers.BatchNormalization()(dconv_1)
    dconv_1 = layers.Activation("relu")(dconv_1)
    dconv_1 = layers.Concatenate(axis=3)([dconv_1, old_deep_layer])
    dconv_1 = layers.Conv2D(f / 2, kernel_size=1, padding="same", use_bias=False)(
        dconv_1
    )
    dconv_1 = layers.BatchNormalization()(dconv_1)
    dconv_1 = layers.Activation("relu")(dconv_1)
    dconv_1 = layers.Activation("relu")(dconv_1)

    return deep_layer(dconv_1, f, k)


def dense_unet(shape, f: int, k: int, batch_size: int, pretrained_weights=None):
    x_input = layers.Input(shape=shape, batch_size=batch_size)

    # -------------------------------------------------------------------------

    l = 1

    conv_1 = layers.Conv2D(f / 2, kernel_size=3, padding="same", use_bias=False)(
        x_input
    )
    conv_1 = layers.BatchNormalization()(conv_1)
    conv_1 = layers.Activation("relu")(conv_1)

    deep_1 = deep_layer(conv_1, f, k)

    # -------------------------------------------------------------------------

    f *= 2
    k *= 2
    l = 2

    conv_2 = layers.MaxPooling2D(pool_size=2, strides=2)(deep_1)
    deep_2 = deep_layer(conv_2, f, k)

    # -------------------------------------------------------------------------

    f *= 2
    k *= 2
    l = 3

    conv_3 = layers.MaxPooling2D(pool_size=2, strides=2)(deep_2)
    deep_3 = deep_layer(conv_3, f, k)

    # -------------------------------------------------------------------------

    f *= 2
    k *= 2
    l = 4

    conv_4 = layers.MaxPooling2D(pool_size=2, strides=2)(deep_3)
    deep_4 = deep_layer(conv_4, f, k)

    # -------------------------------------------------------------------------

    f *= 2
    k *= 2
    l = 5

    conv_5 = layers.MaxPooling2D(pool_size=2, strides=2)(deep_4)
    deep_5 = deep_layer(conv_5, f, k)

    # -------------------------------------------------------------------------

    f /= 2
    k /= 2
    l = 4

    dconv_1 = dconv_layer(deep_5, f, k, deep_4)

    # -------------------------------------------------------------------------

    f /= 2
    k /= 2
    l = 3

    dconv_2 = dconv_layer(dconv_1, f, k, deep_3)

    # -------------------------------------------------------------------------

    f /= 2
    k /= 2
    l = 2

    dconv_3 = dconv_layer(dconv_2, f, k, deep_2)

    # -------------------------------------------------------------------------

    f /= 2
    k /= 2
    l = 1

    dconv_4 = dconv_layer(dconv_3, f, k, deep_1)

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

    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
