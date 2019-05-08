import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, GaussianNoise
from keras.layers.merge import add
from keras import regularizers


def resnet8(img_width, img_height, img_channels, output_dim,
            freeze_filters=False):
    """
    Define model architecture.

    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.

    # Returns
       model: A Model instance.
    """

    # Input
    # Swap width and height because the numpy shape is (rows x cols)
    img_input = Input(shape=(img_height, img_width, img_channels))
    input_noised = GaussianNoise(0.05)(img_input)

    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same',
                trainable=(not freeze_filters))(input_noised)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = keras.layers.normalization.BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(0.001),
                trainable=(not freeze_filters))(x2)

    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(0.001),
                trainable=(not freeze_filters))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same',
                trainable=(not freeze_filters))(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = keras.layers.normalization.BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(0.001),
                trainable=(not freeze_filters))(x4)

    x4 = keras.layers.normalization.BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(0.001),
                trainable=(not freeze_filters))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same',
                trainable=(not freeze_filters))(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = keras.layers.normalization.BatchNormalization()(x5)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(0.001),
                trainable=(not freeze_filters))(x6)

    x6 = keras.layers.normalization.BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(0.001),
                trainable=(not freeze_filters))(x6)

    x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same',
                kernel_regularizer=regularizers.l2(0.001),
                trainable=(not freeze_filters))(x5)
    x7 = add([x5, x6])

    x = Flatten(trainable=(not freeze_filters))(x7)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # h_noise = GaussianNoise(0.01)(x)
    h1 = Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    h1 = Dropout(0.3)(h1)
    h2 = Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.001))(h1)
    h2 = Dropout(0.2)(h2)
    # Gate localization
    localization = Dense(output_dim, activation='softmax')(h2) # Logits + Softmax

    model = Model(inputs=[img_input], outputs=[localization])
    print(model.summary())

    return model
