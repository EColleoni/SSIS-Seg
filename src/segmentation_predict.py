import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
sys.path.insert(1, 'sim2real/utils')

import numpy as np
import pylib as py
import tensorflow as tf
import tqdm
import cv2
import copy
import h5py

import module_spectral as module
from sim2real_segmentation_model import model_Unet_sim2real
from segmentation_models.losses import bce_jaccard_loss

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

def predict(args):

    # ==============================================================================
    # =                                    data                                    =
    # ==============================================================================


    test_image_gen = tf.keras.preprocessing.image.ImageDataGenerator(dtype='float32', rescale=1./255)

    test_img_generator = test_image_gen.flow_from_directory(args.dataset_dir, target_size=args.load_size, 
                                                        color_mode='rgb', class_mode=None, batch_size=args.batch_size, shuffle=True, seed=1, 
                                                        interpolation='bilinear')


    # ==============================================================================
    # =                                   models                                   =
    # ==============================================================================

    segmentation_model = tf.keras.models.load_model(args.model_dir)

    # ==============================================================================
    # =                                  sample                                    =
    # ==============================================================================

    @tf.function
    def test_step(img):
        prediction = segmentation_model(img)
        return prediction

    # ==============================================================================
    # =                                    run                                     =
    # ==============================================================================

    # predict
    len_test_set = len(os.listdir(args.dataset_dir + '/fold'))
    t = tqdm.tqdm(range(int(len_test_set/args.batch_size)))
    idx = 0

    for _ in t:

        img = next(test_img_generator)
        prediction = test_step(img)
        prediction = prediction.numpy()
        
        for n in range(args.batch_size):
            cv2.imwrite(os.path.join(args.save_dir, f'img_{idx}.{args.img_format}'), prediction[n, ] * 255)
            idx += 1


py.arg('--dataset_dir', default='/home/ema/my_workspace/datasets/dataset_simulation/UCL/train/imgs')
py.arg('--model_dir', default='/home/ema/my_workspace/output_seg/models/model_003.h5')
py.arg('--save_dir', default='/home/ema/my_workspace/output_seg/df')
py.arg('--load_size', type=int, default=[512, 512])  # load image to this size
py.arg('--batch_size', type=int, default=8)
py.arg('--img_format', type=str, default='png')

args = py.args()
predict(args)
