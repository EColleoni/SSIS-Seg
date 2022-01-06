import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras import layers
from spectral import SpectralConv2D, SpectralConv2DTranspose
import tensorflow.keras as keras
import numpy as np

n_res = 4
n_down = 2
n_up=2
mlp_dim = 256
dropout_rate = 0.3

def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        # return keras.layers.BatchNormalization
        return InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


def encoder_content(input_shape=(512, 512, 3), n_channels=64, norm='instance_norm'):

    inputs = keras.Input(shape=input_shape)

    Norm = _get_norm_layer(norm)


    def resblock(x_init, channels):

        paddings_1 = tf.constant([[0, 0], [1, 1,], [1, 1], [0, 0]])
        
        x = tf.pad(x_init, paddings_1, 'REFLECT')
        x = SpectralConv2D(filters=channels, kernel_size=3, strides=1, padding='valid')(x)
        x = Norm()(x)
        x = tf.nn.relu(x)

        x = tf.pad(x, paddings_1, 'REFLECT')
        x = SpectralConv2D(filters=channels, kernel_size=3, strides=1, padding='valid')(x)
        x = Norm()(x)

        return x + x_init


    paddings_1 = tf.constant([[0, 0], [1, 1,], [1, 1], [0, 0]])
    paddings_3 = tf.constant([[0, 0], [3, 3,], [3, 3], [0, 0]])

    x = tf.pad(inputs, paddings_3, 'REFLECT')
    x = SpectralConv2D(filters=n_channels, kernel_size=7, strides=1, padding='valid')(x)
    x = Norm()(x)
    x = tf.nn.relu(x)

    for _ in range(n_down) :

        x = tf.pad(x, paddings_1, 'REFLECT')
        x = SpectralConv2D(filters=n_channels*2, kernel_size=4, strides=2, padding='valid')(x)
        x = Norm()(x)
        x = tf.nn.relu(x)

        n_channels = n_channels * 2

    for _ in range(n_res) :
        x = resblock(x, n_channels)

    return keras.Model(inputs=inputs, outputs=x)


def generator(input_shape_content=(32, 32, 256)):

    inputs_content = keras.Input(shape=input_shape_content)

    Norm = _get_norm_layer('instance_norm')

    paddings_1 = tf.constant([[0, 0], [1, 1,], [1, 1], [0, 0]])
    paddings_2 = tf.constant([[0, 0], [2, 2,], [2, 2], [0, 0]])
    paddings_3 = tf.constant([[0, 0], [3, 3,], [3, 3], [0, 0]])

    def resblock(x_init, channels):
    
        paddings_1 = tf.constant([[0, 0], [1, 1,], [1, 1], [0, 0]])
        
        x = tf.pad(x_init, paddings_1, 'REFLECT')
        x = SpectralConv2D(filters=channels, kernel_size=3, strides=1, padding='valid')(x)
        x = Norm()(x)
        x = tf.nn.relu(x)

        x = tf.pad(x, paddings_1, 'REFLECT')
        x = SpectralConv2D(filters=channels, kernel_size=3, strides=1, padding='valid')(x)
        x = Norm()(x)

        return x + x_init

    n_channels = 256
    x = inputs_content

    for i in range(n_res):
        idx = 2 * i
        x = resblock(x, n_channels)
    
    for i in range(n_up):
        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)

        x = tf.pad(x, paddings_2, 'REFLECT')
        x = SpectralConv2D(filters=n_channels//2, kernel_size=5, strides=1, padding='valid')(x)
        x = Norm()(x)
        x = tf.nn.relu(x)

        n_channels = n_channels // 2

    
    x = tf.pad(x, paddings_3, 'REFLECT')
    x = SpectralConv2D(filters=3, kernel_size=7, strides=1, padding='valid')(x)
    x = tf.tanh(x)

    return keras.Model(inputs=inputs_content, outputs=x)


def encoder_cc(input_shape=(256, 256, 3),
                    output_channels=3,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=8,
                    norm='instance_norm',
                    n_att_blocks=1):
    Norm = _get_norm_layer(norm)

    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        h = SpectralConv2D(dim, 3, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        h = SpectralConv2D(dim, 3, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        return keras.layers.add([x, h])

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = SpectralConv2D(dim, 7, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = SpectralConv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 3
    for _ in range(int(n_blocks/2)):
        h = _residual_block(h)

    return keras.Model(inputs=inputs, outputs=h)


def encoder_cc_dropout(input_shape=(256, 256, 3),
                    output_channels=3,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=8,
                    norm='instance_norm',
                    n_att_blocks=1):
    Norm = _get_norm_layer(norm)

    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        h = SpectralConv2D(dim, 3, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        h = SpectralConv2D(dim, 3, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)
        return keras.layers.add([x, h])

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = SpectralConv2D(dim, 7, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)
    h = tf.keras.layers.Dropout(dropout_rate)(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = SpectralConv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 3
    for _ in range(int(n_blocks/2)):
        h = _residual_block(h)

    return keras.Model(inputs=inputs, outputs=h)


def generator_cc(input_shape=(64, 64, 256),
                    output_channels=3,
                    dim=256,
                    n_downsamplings=2,
                    n_blocks=8,
                    norm='instance_norm',
                    n_att_blocks=1):
    Norm = _get_norm_layer(norm)

    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        h = SpectralConv2D(dim, 3, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        h = SpectralConv2D(dim, 3, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        return keras.layers.add([x, h])


    h = inputs = keras.Input(shape=input_shape)

    for _ in range(int(n_blocks/2)):
        h = _residual_block(h)

    # 4
    for _ in range(n_downsamplings):
            
        dim //= 2
        h = SpectralConv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 5
    h = SpectralConv2D(output_channels, 3, padding='same')(h)
    h = tf.tanh(h)

    return keras.Model(inputs=inputs, outputs=h)


def generator_cc_dropout(input_shape=(64, 64, 256),
                    output_channels=3,
                    dim=256,
                    n_downsamplings=2,
                    n_blocks=8,
                    norm='instance_norm',
                    n_att_blocks=1):
    Norm = _get_norm_layer(norm)

    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        h = SpectralConv2D(dim, 3, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        h = SpectralConv2D(dim, 3, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        return keras.layers.add([x, h])


    h = inputs = keras.Input(shape=input_shape)

    for _ in range(int(n_blocks/2)):
        h = _residual_block(h)

    # 4
    for _ in range(n_downsamplings):
            
        dim //= 2
        h = SpectralConv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 5
    h = SpectralConv2D(output_channels, 3, padding='same')(h)
    h = tf.tanh(h)

    return keras.Model(inputs=inputs, outputs=h)


def ConvDiscriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm',
                      n_att_blocks=1):
    dim_ = dim
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = SpectralConv2D(dim, 3, strides=2, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = SpectralConv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)
        #h = _residual_block(h)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = SpectralConv2D(dim, 3, strides=1, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # 3
    h = SpectralConv2D(1, 3, strides=1, padding='same')(h)

    return keras.Model(inputs=inputs, outputs=h)


def ConvDiscriminator_dropout(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm',
                      n_att_blocks=1):
    dim_ = dim
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = SpectralConv2D(dim, 3, strides=2, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)
    h = tf.keras.layers.Dropout(dropout_rate)(h)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = SpectralConv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = SpectralConv2D(dim, 3, strides=1, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # 3
    h = SpectralConv2D(1, 3, strides=1, padding='same')(h)

    return keras.Model(inputs=inputs, outputs=h)


def ConvDiscriminator_no_spectral(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm',
                      n_att_blocks=1):
    dim_ = dim
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = tf.keras.layers.Conv2D(dim, 3, strides=2, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = tf.keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)
        #h = _residual_block(h)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = tf.keras.layers.Conv2D(dim, 3, strides=1, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # 3
    h = tf.keras.layers.Conv2D(1, 3, strides=1, padding='same')(h)

    return keras.Model(inputs=inputs, outputs=h)


def ConvDiscriminator_no_spectral_25(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm',
                      n_att_blocks=1):
    dim_ = dim
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = tf.keras.layers.Conv2D(dim, 3, strides=1, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = tf.keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)
        #h = _residual_block(h)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = tf.keras.layers.Conv2D(dim, 3, strides=1, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # 3
    h = tf.keras.layers.Conv2D(1, 3, strides=1, padding='same')(h)

    return keras.Model(inputs=inputs, outputs=h)


def DCGAN():

    Norm = _get_norm_layer('instance_norm')

    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(128,)))
    model.add(Norm())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 256)
    model.add(Norm())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(Norm())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(Norm())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 128)
    model.add(Norm())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 128, 128, 64)
    model.add(Norm())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 256, 256, 64)
    model.add(Norm())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 512, 512, 3)

    return model


def model_cc(input_shape_enc=(256, 256, 3),
                    shape_dec=(128, 128, 255),
                    output_channels=3,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=8,
                    norm='instance_norm',
                    n_att_blocks=1):

    # 0
    h = inputs = keras.Input(shape=input_shape_enc)

    encoder = encoder_cc(input_shape=input_shape_enc)
    decoder = generator_cc(input_shape=shape_dec)

    h = decoder(encoder(h))
    return keras.Model(inputs=inputs, outputs=h)