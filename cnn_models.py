import keras

from keras import regularizers
from keras.models import Model
from keras.layers.merge import add
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, GaussianNoise, SpatialDropout2D


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
    input_noised = GaussianNoise(0.01)(img_input)

    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same',
                trainable=(not freeze_filters))(input_noised)
    x1 = SpatialDropout2D(0.1)(x1)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = keras.layers.normalization.BatchNormalization()(x1)
    x2 = Activation('relu')(x1)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l1_l2(0.001),
                trainable=(not freeze_filters))(x2)
    x2 = SpatialDropout2D(0.2)(x2)

    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l1_l2(0.001),
                trainable=(not freeze_filters))(x2)
    x2 = SpatialDropout2D(0.2)(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same',
                trainable=(not freeze_filters))(x1)
    x1 = SpatialDropout2D(0.1)(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = keras.layers.normalization.BatchNormalization()(x3)
    x4 = Activation('relu')(x3)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l1_l2(0.001),
                trainable=(not freeze_filters))(x4)
    x4 = SpatialDropout2D(0.4)(x4)

    x4 = keras.layers.normalization.BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l1_l2(0.001),
                trainable=(not freeze_filters))(x4)
    x4 = SpatialDropout2D(0.4)(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same',
                trainable=(not freeze_filters))(x3)
    x3 = SpatialDropout2D(0.3)(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = keras.layers.normalization.BatchNormalization()(x5)
    x6 = Activation('relu')(x5)
    x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l1_l2(0.001),
                trainable=(not freeze_filters))(x6)
    x6 = SpatialDropout2D(0.4)(x6)

    x6 = keras.layers.normalization.BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l1_l2(0.001),
                trainable=(not freeze_filters))(x6)
    x6 = SpatialDropout2D(0.4)(x6)

    x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same',
                kernel_regularizer=regularizers.l1_l2(0.001),
                trainable=(not freeze_filters))(x5)
    x5 = SpatialDropout2D(0.4)(x5)
    x7 = add([x5, x6])

    x = Flatten(trainable=(not freeze_filters))(x7)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # h_noise = GaussianNoise(0.01)(x)
    h1 = Dense(500, activation='relu', kernel_regularizer=regularizers.l1_l2(0.001))(x)
    h1 = Dropout(0.5)(h1)
    h2 = Dense(100, activation='relu', kernel_regularizer=regularizers.l1_l2(0.001))(h1)
    h2 = Dropout(0.5)(h2)
    # Gate localization
    localization = Dense(output_dim, activation='softmax')(x) # Logits + Softmax

    model = Model(inputs=[img_input], outputs=[localization])
    print(model.summary())

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

def mobilenet_v2(img_width, img_height, img_channels, output_dim):
    # Input
    # Swap width and height because the numpy shape is (rows x cols)
    img_input = Input(shape=(img_height, img_width, img_channels))
    input_noised = GaussianNoise(0.05)(img_input)
    model = MobileNetV2(include_top=False, weights='imagenet',
                        input_tensor=img_input, alpha=1.0,
                        input_shape=(img_height, img_width, img_channels))
    x = model.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu',
              kernel_regularizer=regularizers.l1_l2(0.001))(x)
    x = Dropout(0.2)(x)
    predictions = Dense(output_dim, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)
    print(model.summary())

    return model
