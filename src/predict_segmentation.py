import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
from os.path import dirname, abspath
sys.path.insert(1, os.path.join(dirname(dirname(abspath(__file__) )), "utils"))

import pylib as py
import tensorflow as tf
import tqdm
import cv2
import h5py

from utils import create_path


def predict(args):
    """Produce segmentation predictions for all frames in the specified folder.
       
    Args:
        args: list of arguments that specify dataset path and models path.
            
    """

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
        """Return the predicted segmentation masks for the given image batch."""
        
        prediction = segmentation_model(img)
        return prediction

    # ==============================================================================
    # =                                    run                                     =
    # ==============================================================================

    # predict
    len_test_set = len(os.listdir(args.dataset_dir + '/fold'))
    t = tqdm.tqdm(range(int(len_test_set/args.batch_size)))

    create_path(args.save_dir)

    idx = 0

    for _ in t:

        img = next(test_img_generator)
        prediction = test_step(img)
        prediction = prediction.numpy()
        
        for n in range(args.batch_size):
            cv2.imwrite(os.path.join(args.save_dir, f'img_{idx}.{args.img_format}'), prediction[n, ] * 255)
            idx += 1


py.arg('--dataset_dir', default='/home/ema/my_workspace/datasets/dataset_simulation/UCL/train/imgs') # dataset directory
py.arg('--model_dir', default='/home/ema/my_workspace/output_seg/models/model_000.h5') # path to the prediction model
py.arg('--save_dir', default='/home/ema/my_workspace/output_seg/df') # directory where to save data
py.arg('--load_size', type=int, default=[512, 512])  # load image to this size
py.arg('--batch_size', type=int, default=8) # batch size
py.arg('--img_format', type=str, default='png') # format to save synthetic images

args = py.args()
predict(args)
