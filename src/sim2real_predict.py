import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import sys
sys.path.insert(1, 'sim2real/utils')

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tqdm
import cv2

import data
from random import randrange
from sim2real_models import model_cc
from segmentation_models import model_Unet_sim2real



def predict(args):

    # Pick and return a random background image from the specified folder
    def random_background(backgrounds_path):
        list_backgrounds = os.listdir(backgrounds_path)
        background = cv2.imread(os.path.join(backgrounds_path, list_backgrounds[randrange(len(list_backgrounds))]))
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        background = background*(2/255) - 1
        background = tf.image.resize(background, [args.load_size[0], args.load_size[1]])
        background = tf.image.random_crop(background, [args.crop_size, args.crop_size, tf.shape(background)[-1]])
        return tf.expand_dims(background, axis=0)
    
    # Binarize an image using differentiable operations
    def binarize_img(img, thr):
        img = 500*(img - ((2/255)*thr - 1))
        mask = tf.math.sigmoid(img)
        return mask

    # Blend the tools on a random background image from the specified folder
    def background_addition(A):
        background = random_background(args.backgrounds_dir)
        mask = binarize_img(A, 5)
        return tf.math.add(tf.math.multiply(A, mask), tf.math.multiply(background, 1 - mask))

    output_dir = args.save_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ==============================================================================
    # =                                    data                                    =
    # ==============================================================================

    img_paths_test = py.glob(py.join(args.datasets_dir), '*.{}'.format(args.img_format))
    img_paths_test.sort()

    dataset_test, len_dataset = data.make_zip_dataset(img_paths_test, img_paths_test, args.batch_size, args.load_size, args.crop_size, training=False, repeat=False, shuffle=False)


    # ==============================================================================
    # =                                   models                                   =
    # ==============================================================================

    generator_A2B = tf.keras.models.load_model(args.models_dir + '/model_{}/A2B_gen_{}'.format(args.model_number, args.model_number))
    attention_A = tf.keras.models.load_model(args.models_dir + '/model_{}/A_att_{}'.format(args.model_number, args.model_number))


    # ==============================================================================
    # =                                    run                                     =
    # ==============================================================================

    i = 0

    # main loop
    for A, _ in tqdm.tqdm(dataset_test, desc='Inner Epoch Loop', total=len_dataset):
  
        #Predict
        A = background_addition(A)
        A2B_ = generator_A2B(A)
        att_A = attention_A(A)
        att_A = tf.image.grayscale_to_rgb(att_A)
        A2B = tf.math.add(tf.math.multiply(A2B_, att_A), tf.math.multiply(A, 1 - att_A))

        # Save
        A = A.numpy()
        A2B = A2B.numpy()
        att_A = att_A.numpy() * 255
        A = np.uint8((A + 1) * (255/2))
        A2B = np.uint8((A2B + 1) * (255/2))
        
        for n in range(args.batch_size):
            cv2.imwrite(os.path.join(output_dir, f'img-%09d.{args.img_format}' % i), cv2.cvtColor(A2B[n, ], cv2.COLOR_RGB2BGR))
            i += 1

if __name__ == '__main__':
    
    py.arg('--model_number', default='000')
    py.arg('--models_dir', default='/home/ema/my_workspace/output_Westmoreland/models')
    py.arg('--save_dir', default='/home/ema/my_workspace/datasets/df')
    py.arg('--datasets_dir', default='/home/ema/my_workspace/datasets/sim2real/MICCAI_2020/version_1/trainA')
    py.arg('--backgrounds_dir', default='/home/ema/my_workspace/datasets/backgrounds/MICCAI_2017/backgrounds')
    py.arg('--load_size', type=int, default=[576, 720])  # load image to this size
    py.arg('--crop_size', type=int, default=512)  # then crop to this size
    py.arg('--batch_size', type=int, default=8)
    py.arg('--img_format', default='png')
    args = py.args()

    predict(args)

