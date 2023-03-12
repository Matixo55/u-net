from keras.layers import BatchNormalization
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.optimizer_v2.adam import Adam

def deep_layer(input_x, f, k):
    F = f / 2

    flat_1 = layers.Conv2D(f, kernel_size=3, padding="same", activation="relu")(input_x)
    return layers.Conv2D(f, kernel_size=3, padding="same", activation="relu")(flat_1)

    deep_1a = Conv2D(F, kernel_size=3, padding="same")(input_x)
    deep_1 = Conv2D(F, kernel_size=1, padding="same")(deep_1a)
    deep_1 = BatchNormalization()(deep_1)
    deep_1 = Activation("relu")(deep_1)
    deep_1 = Conv2D(k, kernel_size=3, padding="same")(deep_1)
    deep_1 = BatchNormalization()(deep_1)
    deep_1 = Activation("relu")(deep_1)

    deep_2 = Concatenate(axis=3)([deep_1a, deep_1])
    deep_2 = Conv2D(F, kernel_size=1, padding="same")(deep_2)
    deep_2 = BatchNormalization()(deep_2)
    deep_2 = Activation("relu")(deep_2)
    deep_2 = Conv2D(k, kernel_size=3, padding="same")(deep_2)
    deep_2 = BatchNormalization()(deep_2)
    deep_2 = Activation("relu")(deep_2)

    deep_3 = Concatenate(axis=3)([deep_1a, deep_1, deep_2])
    deep_3 = Conv2D(F, kernel_size=1, padding="same")(deep_3)
    deep_3 = BatchNormalization()(deep_3)
    deep_3 = Activation("relu")(deep_3)
    deep_3 = Conv2D(k, kernel_size=3, padding="same")(deep_3)
    deep_3 = BatchNormalization()(deep_3)
    deep_3 = Activation("relu")(deep_3)

    deep_4 = Concatenate(axis=3)([deep_1a, deep_1, deep_2, deep_3])
    deep_4 = Conv2D(F, kernel_size=1, padding="same")(deep_4)
    deep_4 = BatchNormalization()(deep_4)
    deep_4 = Activation("relu")(deep_4)
    deep_4 = Conv2D(k, kernel_size=3, padding="same")(deep_4)
    deep_4 = BatchNormalization()(deep_4)
    deep_4 = Activation("relu")(deep_4)

    return Concatenate(axis=3)([deep_1a, deep_1, deep_2, deep_3, deep_4])


def dconv_layer(input, f, k, old_deep_layer):
    dconv_1 = layers.Conv2DTranspose(f, kernel_size=2, strides=(2, 2))(input)
    # dconv_1 = BatchNormalization()(dconv_1)
    dconv_1 = layers.Activation("relu")(dconv_1)
    dconv_1 = layers.Concatenate(axis=3)([dconv_1, old_deep_layer])

    dconv_flat = layers.Conv2D(f / 2, kernel_size=3, padding="same", activation="relu")(
        dconv_1
    )
    return layers.Conv2D(f / 2, kernel_size=3, padding="same", activation="relu")(dconv_flat)

    dconv_1 = Conv2D(f / 2, kernel_size=1, activation="relu", padding="same")(dconv_1)
    dconv_1 = BatchNormalization()(dconv_1)
    dconv_1 = Activation("relu")(dconv_1)

    return deep_layer(dconv_1, f, k)


def unet(shape, f: int, k: int, batch_size: int, pretrained_weights=None):
    x_input = layers.Input(shape=shape, batch_size=batch_size)

    # -------------------------------------------------------------------------

    l = 1

    # conv_1 = Conv2D(f / 2, kernel_size=3, padding="same")(x_input)
    # conv_1 = BatchNormalization()(conv_1)
    # conv_1 = Activation("relu")(conv_1)

    deep_1 = deep_layer(x_input, f, k)

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
        loss="mean_absolute_error",  # mean_squared_error
        metrics=["accuracy"],
        run_eagerly=True,
    )

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# inputs = Input((572, 572, 3))
#
# conv11 = Conv2D(64, 3, activation='relu')(inputs)
# conv1 = Conv2D(64, 3, activation='relu')(conv11)
# pool1 = MaxPooling2D(pool_size=2)(conv1)
#
# conv22 = Conv2D(128, 3, activation='relu')(pool1)
# conv2 = Conv2D(128, 3, activation='relu')(conv22)
# pool2 = MaxPooling2D(pool_size=2)(conv2)
#
# conv33 = Conv2D(256, 3, activation='relu')(pool2)
# conv3 = Conv2D(256, 3, activation='relu')(conv33)
# pool3 = MaxPooling2D(pool_size=2)(conv3)
#
# conv44 = Conv2D(512, 3, activation='relu')(pool3)
# conv4 = Conv2D(512, 3, activation='relu')(conv44)
# pool4 = MaxPooling2D(pool_size=2)(conv4)
#
# conv55 = Conv2D(1024, 3, activation='relu')(pool4)
# conv5 = Conv2D(1024, 3, activation='relu')(conv55)
#
# up6 = Conv2DTranspose(512, 2, strides=(2, 2), activation='relu',padding="same")(conv5)
# crop_shape_conv_4 = (conv4.shape[1] - up6.shape[1]) // 2
# conv4_cropped = Cropping2D((crop_shape_conv_4, crop_shape_conv_4))(conv4)
# merge6 = Concatenate(axis=3)([up6, conv4_cropped])
# conv66 = Conv2D(512, 3, activation='relu')(merge6)
# conv6 = Conv2D(512, 3, activation='relu')(conv66)
#
# up7 = Conv2DTranspose(256, 2, strides=(2, 2), activation='relu',padding="same")(conv6)
# crop_shape_conv_3 = (conv3.shape[1] - up7.shape[1]) // 2
# conv3_cropped = Cropping2D((crop_shape_conv_3, crop_shape_conv_3))(conv3)
# merge7 = Concatenate(axis=3)([up7, conv3_cropped])
# conv77 = Conv2D(256, 3, activation='relu')(merge7)
# conv7 = Conv2D(256, 3, activation='relu')(conv77)
#
# up8 = Conv2DTranspose(128, 2,strides=(2, 2), activation='relu',padding="same")(conv7)
# crop_shape_conv_2 = (conv2.shape[1] - up8.shape[1]) // 2
# conv2_cropped = Cropping2D((crop_shape_conv_2, crop_shape_conv_2))(conv2)
# merge8 = Concatenate(axis=3)([up8, conv2_cropped])
# conv88 = Conv2D(128, 3, activation='relu')(merge8)
# conv8 = Conv2D(128, 3, activation='relu')(conv88)
#
# up9 = Conv2DTranspose(64, 2,strides=(2, 2), activation='relu',padding="same")(conv8)
# crop_shape_conv_1 = (conv1.shape[1] - up9.shape[1]) // 2
# conv1_cropped = Cropping2D((crop_shape_conv_1, crop_shape_conv_1))(conv1)
# merge9 = Concatenate(axis=3)([up9, conv1_cropped])
# conv99 = Conv2D(64, 3, activation='relu')(merge9)
# conv9 = Conv2D(64, 3, activation='relu')(conv99)
# conv10 = Conv2D(3, 1, activation='relu')(conv9)
#
# model = Model(inputs=inputs, outputs=conv10)
#
# model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
# x_input = Input(shape=(572, 572, 3), batch_size=8)
# # -------------------------------------------------------------------------
# conv_1 = Conv2D(64, 3, activation='relu')(x_input)
# conv_2 = Conv2D(64, 3, activation='relu')(conv_1)
# pool_1 = MaxPooling2D(pool_size=2, strides=2)(conv_2)
# # -------------------------------------------------------------------------
# conv_3 = Conv2D(128, 3, activation='relu')(pool_1)
# conv_4 = Conv2D(128, 3, activation='relu')(conv_3)
# pool_2 = MaxPooling2D(pool_size=2, strides=2)(conv_4)
# # -------------------------------------------------------------------------
# conv_5 = Conv2D(256, 3, activation='relu')(pool_2)
#
# conv_6 = Conv2D(256, 3, activation='relu')(conv_5)
#
# pool_3 = MaxPooling2D(pool_size=2, strides=2)(conv_6)
# # -------------------------------------------------------------------------
# conv_7 = Conv2D(512, 3, activation='relu')(pool_3)
# conv_8 = Conv2D(512, 3, activation='relu')(conv_7)
#
# pool_4 = MaxPooling2D(pool_size=2, strides=2)(conv_8)
# # -------------------------------------------------------------------------
# conv_9 = Conv2D(1024, 3, activation='relu')(pool_4)
# conv_10 = Conv2D(1024, 3, activation='relu')(conv_9)
#
# def get_cropping_shape(previous_layer_shape, current_layer_shape):
#     return int((previous_layer_shape - current_layer_shape) / 2)
#
# up_conv_10a = Conv2DTranspose(512, 2, strides=(2, 2))(conv_10)
# crop_shape_conv_8 = get_cropping_shape(conv_8.shape[1], up_conv_10a.shape[1])
# conv_8_cropped = Cropping2D((crop_shape_conv_8, crop_shape_conv_8))(conv_8)
# up_conv_10b = Concatenate(axis=3)([up_conv_10a, conv_8_cropped])
# conv_11 = Conv2D(512, 3, activation='relu')(up_conv_10b)
# conv_12 = Conv2D(512, 3, activation='relu')(conv_11)
# # # -------------------------------------------------------------------------
# up_conv_13a = Conv2DTranspose(256, 2, strides=(2, 2))(conv_12)
# crop_shape_conv_6 = get_cropping_shape(conv_6.shape[1], up_conv_13a.shape[1])
# conv_6_cropped = Cropping2D((crop_shape_conv_6, crop_shape_conv_6))(conv_6)
# up_conv_13b = Concatenate(axis=3)([up_conv_13a, conv_6_cropped])
# conv_14 = Conv2D(256, 3, activation='relu')(up_conv_13b)
# conv_15 = Conv2D(256, 3, activation='relu')(conv_14)
# # -------------------------------------------------------------------------
# up_conv_16a = Conv2DTranspose(128, 2, strides=(2, 2))(conv_15)
# crop_shape_conv_4 = get_cropping_shape(conv_4.shape[1], up_conv_16a.shape[1])
# conv_4_cropped = Cropping2D((crop_shape_conv_4, crop_shape_conv_4))(conv_4)
# up_conv_16b = Concatenate(axis=3)([up_conv_16a, conv_4_cropped])
# conv_17 = Conv2D(128, 3, activation='relu')(up_conv_16b)
# conv_18 = Conv2D(128, 3, activation='relu')(conv_17)
# # -------------------------------------------------------------------------
# up_conv_19a = Conv2DTranspose(64, 2, strides=(2, 2))(conv_18)
# crop_shape_conv_2 = get_cropping_shape(conv_2.shape[1], up_conv_19a.shape[1])
# conv_2_cropped = Cropping2D((crop_shape_conv_2, crop_shape_conv_2))(conv_2)
# up_conv_19b = Concatenate(axis=3)([up_conv_19a, conv_2_cropped])
# conv_19 = Conv2D(64, 3, activation='relu')(up_conv_19b)
# conv_20 = Conv2D(64, 3, activation='relu')(conv_19)
# out = Conv2D(3, 1, activation='softmax')(conv_20)
