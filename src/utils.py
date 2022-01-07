import os
import cv2
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import shape
import tensorflow_addons as tfa

from random import randrange, randint

# Useful functions used across training and testing scripts

def manage_batch_size_tf(func):
    """Wrapper to run a function that was designed for an
        input with batch size 1 over all Tensors.
    """

    def wrapper(input, *args):
        input_shape = shape(input)
        input_shape = input_shape.numpy()
        assert len(input_shape) == 4

        if input_shape[0] == 1:
            return func(input, *args)
        
        elif input_shape[0] > 1:
            results = [func(tf.expand_dims(input[n], axis=0), *args) for n in range(input_shape[0])]
            results = list(map(list, zip(*results)))
            results = [tf.stack(results[n]) for n in range(len(results))]

            if len(results) > 1:
                return tuple(results)
            else:
                return results[0]

        else:
            print("Invalid shape for the input")

    return wrapper

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def random_background(backgrounds_path, args):    
    """Picks a random background from the specified
        path and return it as a tensor.

    Args:
        backgrounds_path: path to the directory containing 
            surgical backgorund images.

    Returns:
        Tensor containing a surgical background image.

    """

    list_backgrounds = os.listdir(backgrounds_path)
    background = cv2.imread(os.path.join(backgrounds_path, list_backgrounds[randrange(len(list_backgrounds))]))
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    background = background*(2/255) - 1
    background = tf.image.resize(background, [args.load_size[0], args.load_size[1]])
    background = tf.image.random_crop(background, [args.crop_size, args.crop_size, tf.shape(background)[-1]])
    return tf.expand_dims(background, axis=0)

def binarize_img(img, thr):
    """Binarizes img using thr in range [0, 255] --> [-1, 1]
        All operations are differentiable to support backpropagation.

    Args:
        img: Tensor image to be binarized.

        thr: threshold in range [0, 255] to binarize the image.

    Returns:
        Binarized tensor image in range [-1, 1]

    """

    img = 500*(img - ((2/255)*thr - 1))
    mask = tf.math.sigmoid(img)
    return mask

def binarize_mask(img, thr):
    """Binarizes masks using thr in range [0, 255] --> [0, 1].
        All operations are differentiable to support backpropagation.

    Args:
        img: Tensor grayscale mask to be binarized.

        thr: threshold in range [0, 255] to binarize the mask.

    Returns:
        Binarized tensor mask in range [0, 1]

    """

    img = 500*(img - ((1/255)*(255 + thr) - 1))
    mask = tf.math.sigmoid(img)
    return mask

@manage_batch_size_tf
def random_flip(A):
    """Applies vertcal flip wit 50% probability and horizontal flip 
        with probability 10%.

    Args:
        A: Tensor image.

    Returns:
        Flipped tensor image.

    """

    if randint(0, 1):
        A = tf.image.flip_left_right(A)
    if randint(0, 9) == 9:
        A = tf.image.flip_up_down(A)
    return A

@manage_batch_size_tf
def background_addition(A, args):
    """Blends tools A on a random background.

    Args:
        A: Tensor tools image with black backgound.

    Returns:
        Tensor image with tools blended on a surgical background.

    """

    background = random_background(args.backgrounds_dir, args)
    mask = binarize_img(A, 5)
    return tf.math.add(tf.math.multiply(A, mask), tf.math.multiply(background, 1 - mask))

def generate_masks(foreground_mask, args):
    """Produces neighbor and backgrund masks from a tools mask.

    Args:
        foreground_mask: binary tensor mask.

    Returns:
        neighbor_mask: binary tensor mask of the neighbors of the input mask.

        background_mask: binary tensor mask of the background of the input mask.

    """

    filtered_mask = tfa.image.mean_filter2d(foreground_mask, args.neighbor_kernel_size, 'SYMMETRIC')
    background_mask = 1 - binarize_mask(filtered_mask, 5)
    neighbor_mask = 1 - background_mask - foreground_mask
    return neighbor_mask, background_mask