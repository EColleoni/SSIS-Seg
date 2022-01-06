import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
sys.path.insert(1, '/home/ema/my_workspace/codes/SSIS-Seg/utils')

import numpy as np
import pylib as py
import tensorflow as tf
import tqdm
import cv2
import copy
import h5py

import module_spectral as module
from models_segmentation import model_Unet_sim2real
from segmentation_models.losses import bce_jaccard_loss

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

def train(args):
    """Train a segmentation model.
    Args:
        args: list of arguments that specify datasets
            path and hyperparameters.
    """

    # output_dir
    output_dir = py.join('output_seg')
    py.mkdir(output_dir)

    models_path = py.join(output_dir, 'models')
    py.mkdir(models_path)

    # save settings
    py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

    # ==============================================================================
    # =                                    metrics                                 =
    # ==============================================================================

    def calc_IoU(mask_1, mask_2):
        """Calculate Intersection over Union score between two masks.
    
        Args:
            mask_1, mask_2: masks used to evaluate IoU.
        Returns:
            Intersection over union score.
        """
    
        mask_1 = mask_1 > 0.3
        mask_2 = mask_2 > 0.3

        TP = mask_1 * mask_2
        TP = TP.sum()

        FP = ((mask_1 * (1 - mask_2)) + ((1 - mask_1) * mask_2)) * mask_2
        FP = FP.sum()

        FN = ((mask_1 * (1 - mask_2)) + ((1 - mask_1) * mask_2)) * mask_1
        FN = FN.sum()

        return TP / (TP + FP + FN)

    # ==============================================================================
    # =                                    data                                    =
    # ==============================================================================

    def train_gen():
        """Produce a generator for the training set."""

        train_image_gen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True, dtype='float32', rescale=1./255)
        train_gt_gen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True, dtype='float32', rescale=1./255)

        train_img_generator = train_image_gen.flow_from_directory(os.path.join(args.dataset_dir, 'train', 'imgs/'), target_size=args.load_size, 
                                                            color_mode='rgb', class_mode=None, batch_size=args.batch_size, shuffle=True, seed=1, 
                                                            interpolation='bilinear')

        train_gt_generator = train_gt_gen.flow_from_directory(os.path.join(args.dataset_dir, 'train', 'gt_binary/'), target_size=args.load_size, 
                                                            color_mode='grayscale', class_mode=None, batch_size=args.batch_size, shuffle=True, seed=1,
                                                            interpolation='nearest')
                
        while True:
            img_gen = train_img_generator.next()
            gt_gen = train_gt_generator.next()
            yield img_gen, gt_gen

    def val_gen():
        """Produce a generator for the validation set."""

        val_image_gen = tf.keras.preprocessing.image.ImageDataGenerator(dtype='float32', rescale=1./255)
        val_gt_gen = tf.keras.preprocessing.image.ImageDataGenerator(dtype='float32', rescale=1./255)

        val_img_generator = val_image_gen.flow_from_directory(os.path.join(args.dataset_dir, 'validation', 'imgs/'), target_size=args.load_size, 
                                                            color_mode='rgb', class_mode=None, batch_size=1, shuffle=True, seed=1, 
                                                            interpolation='bilinear')

        val_gt_generator = val_gt_gen.flow_from_directory(os.path.join(args.dataset_dir, 'validation', 'gt_binary/'), target_size=args.load_size, 
                                                            color_mode='grayscale', class_mode=None, batch_size=1, shuffle=True, seed=1,
                                                            interpolation='nearest')
                
        while True:
            img_gen = val_img_generator.next()
            gt_gen = val_gt_generator.next()
            yield img_gen, gt_gen


    # ==============================================================================
    # =                                   models                                   =
    # ==============================================================================

    len_train_set = len(os.listdir(os.path.join(args.dataset_dir, 'train', 'imgs', 'fold')))
    len_val_set = len(os.listdir(os.path.join(args.dataset_dir, 'validation', 'imgs', 'fold')))

    segmentation_model = model_Unet_sim2real()

    lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_train_set, args.epoch_decay * len_train_set)

    model_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler, beta_1=args.beta_1)

    # ==============================================================================
    # =                                 train step                                 =
    # ==============================================================================


    @tf.function
    def train_step(img, gt):
        """Trains the segmentation model for one sep.
        
        Args:
            img, gt: images batch and relative ground truth.
                
        Returns:
            Return the segmentation loss for the given batch.
     
        """

        img_ = tf.convert_to_tensor(img)
        gt_ = tf.convert_to_tensor(gt)
        with tf.GradientTape() as t_:

            prediction = segmentation_model(img_, training=True)

            loss = bce_jaccard_loss(gt_, prediction)

        model_grad = t_.gradient(loss, segmentation_model.trainable_variables)

        model_optimizer.apply_gradients(zip(model_grad, segmentation_model.trainable_variables))

        return loss


    # ==============================================================================
    # =                                  sample                                    =
    # ==============================================================================

    @tf.function
    def sample(img):
        """Predicts the segmentation map of a given batch.
        
        Args:
            img: image batch that will be segmented.
                
        Returns:
            Returns the computedprediction for the batch.
                
        """
    
        prediction = segmentation_model(img)
        return prediction

    # ==============================================================================
    # =                                    run                                     =
    # ==============================================================================

    t_generator = train_gen()
    v_generator = val_gen()

    best_val_metric = np.NINF

    # epoch counter
    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

    # summary
    train_summary_writer = tf.summary.create_file_writer(os.path.join(output_dir, 'summaries', 'train'))

    # sample
    sample_dir = py.join(output_dir, 'samples_training')
    py.mkdir(sample_dir)

    # main loop
    with train_summary_writer.as_default():
        for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):

            # update epoch counter
            ep_cnt.assign_add(1)

            # train for an epoch
            
            t = tqdm.tqdm(range(int(len_train_set/args.batch_size)))

            for _ in t:
                img, gt = next(t_generator)

                loss = train_step(img, gt)
                t.set_description("Training loss: {} |".format(str(np.array(loss))))
            
            # Compute loss and metrics on the validation set
            val_metric = []
            for _ in range(len_val_set):
                img, gt = next(v_generator)
                prediction = sample(img).numpy()
                val_metric.append(calc_IoU(prediction, gt))
            
            val_metric = np.mean(np.asarray(val_metric))

            img = np.concatenate((img[0], cv2.cvtColor(prediction[0],cv2.COLOR_GRAY2RGB), cv2.cvtColor(gt[0],cv2.COLOR_GRAY2RGB)), axis=1) * 255
            cv2.imwrite(py.join(sample_dir, 'epoch_{}-iter_{}.jpg'.format(str(ep).zfill(3), str(model_optimizer.iterations.numpy()).zfill(9))), img)

            # Print validation loss and save the model if the validation metric improved.
            if val_metric >= best_val_metric:
                print('Validation metric improved from {} to {}'.format(str(best_val_metric), str(val_metric)))
                best_val_metric = copy.deepcopy(val_metric)
                segmentation_model.save(py.join(models_path, 'model_{}.h5'.format(str(ep).zfill(3))), overwrite=True, save_format='h5')
            else:
                print('Validation metric did not improve from {}'.format(str(best_val_metric)))

py.arg('--dataset_dir', default='/home/ema/my_workspace/datasets/dataset_simulation/UCL') # Dataset directory
py.arg('--load_size', type=int, default=[512, 512])  # load image to this size
py.arg('--batch_size', type=int, default=8) # Batch size
py.arg('--epochs', type=int, default=100) # # number of epochs
py.arg('--epoch_decay', type=int, default=25)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0001) # learning rate
py.arg('--beta_1', type=float, default=0.9) # beta 1 value to setup Adam optimizer
args = py.args()
train(args)

