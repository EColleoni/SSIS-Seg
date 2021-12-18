import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization
from spectral import SpectralConv2D, SpectralConv2DTranspose
from attention import SelfAttnModel
import tensorflow.keras as keras


# ==============================================================================
# =                                  networks                                  =
# ==============================================================================

def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


def ResnetGenerator(input_shape=(256, 256, 3),
                    output_channels=3,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=9,
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
    for _ in range(n_blocks):
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


def ConvDiscriminator_SR(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm',
                      n_att_blocks=1):
    dim_ = dim
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
    h = SpectralConv2D(dim, 3, strides=2, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings + 2):
        dim = min(dim * 2, dim_ * 8)
        h = SpectralConv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)
        h = _residual_block(h)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = SpectralConv2D(dim, 3, strides=1, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # 3
    h = SpectralConv2D(1, 3, strides=1, padding='same')(h)

    return keras.Model(inputs=inputs, outputs=h)


# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================

class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate

class StepDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: zero

    def __init__(self, initial_learning_rate, step_decay):
        super(StepDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * 0,
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate
