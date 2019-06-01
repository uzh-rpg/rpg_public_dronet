import keras

from keras import regularizers
from keras.models import Model
from keras.layers.merge import add
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, GaussianNoise, SpatialDropout2D


def resnet8(img_width, img_height, img_channels, output_dim,
            freeze_filters=False, higher_l2=False, hidden_dropout=False):
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
    l2_factor = 1e-4

    if higher_l2:
        l2_factor = 1e-3

    # Input
    # Swap width and height because the numpy shape is (rows x cols)
    img_input = Input(shape=(img_height, img_width, img_channels))
    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same',
                trainable=(not freeze_filters))(img_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = keras.layers.normalization.BatchNormalization()(x1)
    x2 = Activation('relu')(x1)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(l2_factor),
                trainable=(not freeze_filters))(x2)

    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(l2_factor),
                trainable=(not freeze_filters))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same',
                trainable=(not freeze_filters))(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = keras.layers.normalization.BatchNormalization()(x3)
    x4 = Activation('relu')(x3)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(l2_factor),
                trainable=(not freeze_filters))(x4)

    x4 = keras.layers.normalization.BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(l2_factor),
                trainable=(not freeze_filters))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same',
                trainable=(not freeze_filters))(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = keras.layers.normalization.BatchNormalization()(x5)
    x6 = Activation('relu')(x5)
    x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(l2_factor),
                trainable=(not freeze_filters))(x6)

    x6 = keras.layers.normalization.BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(l2_factor),
                trainable=(not freeze_filters))(x6)

    x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same',
                kernel_regularizer=regularizers.l2(l2_factor),
                trainable=(not freeze_filters))(x5)
    x7 = add([x5, x6])

    x = Flatten(trainable=(not freeze_filters))(x7)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    h1 = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(l2_factor))(x)
    h1 = Dropout(0.5)(h1)

    final = x
    if hidden_dropout:
        final = h1
    # Gate localization
    localization = Dense(output_dim, activation='softmax')(final) # Logits + Softmax

    model = Model(inputs=[img_input], outputs=[localization])
    # print(model.summary())

    return model

def resnet50(img_width, img_height, img_channels, output_dim):
    # Input
    # Swap width and height because the numpy shape is (rows x cols)
    img_input = Input(shape=(img_height, img_width, img_channels))
    model = ResNet50(include_top=False, weights='imagenet',
                     input_tensor=img_input, input_shape=(img_height, img_width,
                                                         img_channels),
                     pooling='max')
    x = model.output
    predictions = Dense(output_dim, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)
    print(model.summary())

    return model

def mobilenet_v2(img_width, img_height, img_channels, output_dim, alpha,
                 hidden_dropout, pooling, dropout):
    # Input
    # Swap width and height because the numpy shape is (rows x cols)
    img_input = Input(shape=(img_height, img_width, img_channels))
    model = MobileNetV2(include_top=False, weights='imagenet',
                        input_tensor=img_input, alpha=alpha,
                        input_shape=(img_height, img_width, img_channels),
                        pooling=pooling)
    x = model.output
    if pooling is None:
        x = Flatten()(x)

    if hidden_dropout:
        x = Dropout(0.5)(x)
        x = Dense(100, activation='relu',
                  kernel_regularizer=regularizers.l1_l2(0.001))(x)
        x = Dropout(0.2)(x)
    elif dropout:
        x = Dropout(0.5)(x)

    predictions = Dense(output_dim, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)
    print(model.summary())

    return model
