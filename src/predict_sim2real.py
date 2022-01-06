import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import sys
sys.path.insert(1, '/home/ema/my_workspace/codes/SSIS-Seg/utils')

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tqdm
import cv2

import data
from utils import *


def predict(args):
    """Produce a synthetic frame starting from simulation tools
        blended on a random surgical background.
       
    Args:
        args: list of arguments that specify dataset path, backgrounds
            path and models path.
            
    """

    output_dir = args.save_dir
    create_path(output_dir)

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
        A = background_addition(A, args)
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
    
    py.arg('--models_dir', default='/home/ema/my_workspace/output_Westmoreland/models') # directory containing trained models
    py.arg('--model_number', default='000') # model number from the model's directory
    py.arg('--save_dir', default='/home/ema/my_workspace/datasets/df') # save directory
    py.arg('--datasets_dir', default='/home/ema/my_workspace/datasets/sim2real/MICCAI_2020/version_1/trainA') # dataset directory
    py.arg('--backgrounds_dir', default='/home/ema/my_workspace/datasets/backgrounds/MICCAI_2017/backgrounds') # backgrounds directory
    py.arg('--load_size', type=int, default=[576, 720])  # load image to this size
    py.arg('--crop_size', type=int, default=512)  # then crop to this size
    py.arg('--batch_size', type=int, default=8) # batch size
    py.arg('--img_format', default='png') # format to save synthetic images
    args = py.args()

    predict(args)

