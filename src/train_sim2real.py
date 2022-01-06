import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import sys
from os.path import dirname, abspath
sys.path.insert(1, os.path.join(dirname(dirname(abspath(__file__) )), "utils"))

import functools
import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm
import cv2

import data
import module_spectral as module
from segmentation_models.losses import bce_jaccard_loss as attention_loss_fn
from models_sim2real import model_cc, ConvDiscriminator, ConvDiscriminator_no_spectral, ConvDiscriminator_no_spectral_25, ConvDiscriminator_dropout
from models_segmentation import model_Unet_sim2real
from utils import *

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

def train(args):
    """Train an image-to-image translation model for
       simulation-to-real surgical image sythesis.

    Args:
        args: list of arguments that specify datasets
            path and hyperparameters.

    """

    # output_dir
    output_dir = py.join(args.save_dir, 'output_Westmoreland')
    py.mkdir(output_dir)

    if args.start_saving_epoch is not None:

        models_path = py.join(output_dir, 'models')

        py.mkdir(models_path)

    # save settings
    py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)
    
    args.adversarial_loss_weight = np.linspace(args.adversarial_loss_weight[0], args.adversarial_loss_weight[1], args.epochs)
    args.cycle_loss_weight = np.linspace(args.cycle_loss_weight[0], args.cycle_loss_weight[1], args.epochs)
    args.ssim_loss_weight = np.linspace(args.ssim_loss_weight[0], args.ssim_loss_weight[1], args.epochs)

    # ==============================================================================
    # =                                    data                                    =
    # ==============================================================================

    A_img_paths = py.glob(py.join(args.datasets_dir, 'trainE'), '*.png')
    B_img_paths = py.glob(py.join(args.datasets_dir, 'trainB'), '*.png')
    A_B_dataset, len_dataset = data.make_zip_dataset(A_img_paths, B_img_paths, args.batch_size, args.load_size, args.crop_size, training=True, repeat=False)

    A2B_pool = data.ItemPool(args.pool_size)
    B2A_pool = data.ItemPool(args.pool_size)

    A_img_paths_test = py.glob(py.join(args.datasets_dir, 'trainE'), '*.png')
    B_img_paths_test = py.glob(py.join(args.datasets_dir, 'trainB'), '*.png')
    A_B_dataset_test, _ = data.make_zip_dataset(A_img_paths_test, B_img_paths_test, args.batch_size, args.load_size, args.crop_size, training=False, repeat=True)

    # ==============================================================================
    # =                                   models                                   =
    # ==============================================================================
    
    # Build the generators
    generator_A2B = model_cc(input_shape_enc=(512, 512, 3), shape_dec=(128, 128, 256))
    generator_B2A = model_cc(input_shape_enc=(512, 512, 3), shape_dec=(128, 128, 256))
    
    # Attention module: try to import model weights. If it fails, try to import full model.
    # If it fails again, create a model with no pre-trained weights.
    try:
        attention_A = model_Unet_sim2real(input_shape=(512, 512, 3))
        attention_B = model_Unet_sim2real(input_shape=(512, 512, 3))

        attention_A.load_weights(args.seg_model_path)
        attention_B.load_weights(args.seg_model_path)
    
    except:
        
        try:
            attention_A = tf.keras.models.load_model(args.seg_model_path)
            attention_B = tf.keras.models.load_model(args.seg_model_path)
        
        except:
            attention_A = model_Unet_sim2real(input_shape=(512, 512, 3))
            attention_B = model_Unet_sim2real(input_shape=(512, 512, 3))
    
    # Build the dicriminator
    if args.discriminator == 'ConvDiscriminator':
        D_A = ConvDiscriminator(input_shape=(512, 512, 3))
        D_B = ConvDiscriminator(input_shape=(512, 512, 3))
    elif args.discriminator == 'ConvDiscriminator_no_spectral':
        D_A = ConvDiscriminator_no_spectral(input_shape=(512, 512, 3))
        D_B = ConvDiscriminator_no_spectral(input_shape=(512, 512, 3))
    elif args.discriminator == 'ConvDiscriminator_no_spectral_25':
        D_A = ConvDiscriminator_no_spectral_25(input_shape=(512, 512, 3))
        D_B = ConvDiscriminator_no_spectral_25(input_shape=(512, 512, 3))
    elif args.discriminator == 'ConvDiscriminator_dropout':
        D_A = ConvDiscriminator_dropout(input_shape=(512, 512, 3))
        D_B = ConvDiscriminator_dropout(input_shape=(512, 512, 3))
    
    # Losses definition
    d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
    reconstruction_loss_fn = tf.losses.MeanAbsoluteError()

    # Learning rate definition
    G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
    G_lr_scheduler_att = module.LinearDecay(args.lr_att, args.epochs * len_dataset, args.epoch_decay * len_dataset)
    D_lr_scheduler = module.LinearDecay(args.lr * args.lrD_on_lrG, args.epochs * len_dataset, args.epoch_decay * len_dataset)

    # Optimizers definition
    G_optimizer_A2B_gen = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
    G_optimizer_A_att = keras.optimizers.Adam(learning_rate=G_lr_scheduler_att, beta_1=args.beta_1)

    G_optimizer_B2A_gen = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
    G_optimizer_B_att = keras.optimizers.Adam(learning_rate=G_lr_scheduler_att, beta_1=args.beta_1)

    D_optimizer_A = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)
    D_optimizer_B = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)

    # ==============================================================================
    # =                                 train step                                 =
    # ==============================================================================

    @tf.function
    def train_G(A, B, foreground_A, ep):
        """Train the generatrs and the attention
           modules for one step.

        Args:
            A: image with simulation tools blended on a
                surgical background.
            
            B: real surgical image.

            foreground_A: foreground mask associated to A.

            ep: epoch number.

        Returns:
            A2B: synthetic simulation-to-real image

            B2A: synthetic real-to-simulation image

    """

        with tf.GradientTape() as t_A, tf.GradientTape() as t_B, tf.GradientTape() as t_C, tf.GradientTape() as t_D:

            ############################
            #      image_synthesis     #
            ############################
            
            # Apply image-to-image translation on the inputs
            A2B_ = generator_A2B(A)
            B2A_ = generator_B2A(B)

            # Compute the attention maps for the input 
            att_A = attention_A(A)
            att_B = attention_B(B)
            
            # Compute SS loss
            _, _, ss_loss_A = sim_segmentation_loss(foreground_A, att_A)

            att_A = tf.image.grayscale_to_rgb(att_A)
            att_B = tf.image.grayscale_to_rgb(att_B)

            # Compute half-cycle outputs
            A2B = tf.math.add(tf.math.multiply(A2B_, att_A), tf.math.multiply(A, 1 - att_A))
            B2A = tf.math.add(tf.math.multiply(B2A_, att_B), tf.math.multiply(B, 1 - att_B))

            # Repeat previous operations to close the cycle
            A2B2A_ = generator_B2A(A2B)
            B2A2B_ = generator_A2B(B2A)

            att_A2B = attention_B(A2B)
            att_B2A = attention_A(B2A)

            att_A2B = tf.image.grayscale_to_rgb(att_A2B)
            att_B2A = tf.image.grayscale_to_rgb(att_B2A)

            A2B2A = tf.math.add(tf.math.multiply(A2B2A_, att_A2B), tf.math.multiply(A2B, 1 - att_A2B))
            B2A2B = tf.math.add(tf.math.multiply(B2A2B_, att_B2A), tf.math.multiply(B2A, 1 - att_B2A))

            # Compute discriminator's output
            A2B_d_logits = D_B(A2B, training=True)
            B2A_d_logits = D_A(B2A, training=True)

            ############################
            #     loss_computation     #
            ############################

            # Compute losses
            A2B_adversarial_loss = g_loss_fn(A2B_d_logits)
            B2A_adversarial_loss = g_loss_fn(B2A_d_logits)

            A2B2A_cycle_loss = reconstruction_loss_fn(A, A2B2A)
            B2A2B_cycle_loss = reconstruction_loss_fn(B, B2A2B)

            ssim_A2B2A_loss = 2 - tf.image.ssim(A2B2A, A, max_val=2)
            ssim_B2A2B_loss = 2 - tf.image.ssim(B2A2B, B, max_val=2)

            att_A_loss = attention_loss_fn(att_A, att_A2B)
            att_B_loss = attention_loss_fn(att_B, att_B2A)

            # Merge losses for generators and attention modules
            G_loss_A = args.adversarial_loss_weight[ep] * (A2B_adversarial_loss + B2A_adversarial_loss) + \
                    (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight[ep] + \
                    (ssim_A2B2A_loss + ssim_B2A2B_loss) * args.ssim_loss_weight[ep]

            G_loss_B = args.adversarial_loss_weight[ep] * (A2B_adversarial_loss + B2A_adversarial_loss) + \
                    (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight[ep] + \
                    (ssim_A2B2A_loss + ssim_B2A2B_loss) * args.ssim_loss_weight[ep]

            G_loss_att_A = args.adversarial_loss_weight[ep] * (A2B_adversarial_loss + B2A_adversarial_loss) + \
                    (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight[ep] + \
                    (ssim_A2B2A_loss + ssim_B2A2B_loss) * args.ssim_loss_weight[ep] + \
                    ss_loss_A * args.sim_supervision_weight

            G_loss_att_B = args.adversarial_loss_weight[ep] * (A2B_adversarial_loss + B2A_adversarial_loss) + \
                    (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight[ep] + \
                    (ssim_A2B2A_loss + ssim_B2A2B_loss) * args.ssim_loss_weight[ep] + \
                    (att_A_loss + att_B_loss) * args.attention_weight

        # Compute gradients
        G_grad_generator_A2B = t_A.gradient(G_loss_A, generator_A2B.trainable_variables)
        G_grad_att_A = t_B.gradient(G_loss_att_A, attention_A.trainable_variables)

        G_grad_generator_B2A = t_C.gradient(G_loss_B, generator_B2A.trainable_variables)
        G_grad_att_B = t_D.gradient(G_loss_att_B, attention_B.trainable_variables)

        # Apply backpropagation
        G_optimizer_A2B_gen.apply_gradients(zip(G_grad_generator_A2B, generator_A2B.trainable_variables))
        G_optimizer_A_att.apply_gradients(zip(G_grad_att_A, attention_A.trainable_variables))

        G_optimizer_B2A_gen.apply_gradients(zip(G_grad_generator_B2A, generator_B2A.trainable_variables))
        G_optimizer_B_att.apply_gradients(zip(G_grad_att_B, attention_B.trainable_variables))

        return A2B, B2A

   
    @tf.function
    def train_D(A, B, A2B, B2A, ep):
        """Train the discriminatos for one step.

            Args:
                A: image with simulation tools blended on a
                    surgical background.
                
                B: real surgical image.

                A2B: synthetic simulation-to-real image from
                    the pool.

                B2A: synthetic real-to-simulation image from
                    the pool.

                ep: epoch number.

        """

        with tf.GradientTape() as t_A, tf.GradientTape() as t_B:
            
            # Compute discriminator's output
            A_d_logits = D_A(A, training=True)
            B2A_d_logits = D_A(B2A, training=True)
            B_d_logits = D_B(B, training=True)
            A2B_d_logits = D_B(A2B, training=True)

            # Compute losses
            A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
            B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
            
            # Apply gradient penalty
            D_A_gp = gan.gradient_penalty(functools.partial(D_A, training=True), A, B2A, mode=args.gradient_penalty_mode)
            D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=True), B, A2B, mode=args.gradient_penalty_mode)

            # Merge losses
            D_loss_A = (A_d_loss + B2A_d_loss) * args.adversarial_loss_weight[ep] + D_A_gp * args.gradient_penalty_weight
            D_loss_B = (B_d_loss + A2B_d_loss) * args.adversarial_loss_weight[ep] + D_B_gp * args.gradient_penalty_weight

        # Compute gradients
        D_grad_A = t_A.gradient(D_loss_A, D_A.trainable_variables)
        D_grad_B = t_B.gradient(D_loss_B, D_B.trainable_variables)

        # Apply backpropagation
        D_optimizer_A.apply_gradients(zip(D_grad_A, D_A.trainable_variables))
        D_optimizer_B.apply_gradients(zip(D_grad_B, D_B.trainable_variables))

        return
    
    # Compute SS loss
    def sim_segmentation_loss(foreground_A, att_A):
        """Computes SS loss.

        Args:
            foreground_mask: binary tensor mask.

            att_A: grayscale tensor from the attention modules.

        Returns:
            Scalar number crresponding to the SS loss.

        """
        
        # SS Loss with constrained background
        if args.attention_background_constraint:
            neighbor_A, background_A = generate_masks(foreground_A, args)
            bin_mask = binarize_mask(att_A, args.ss_attention_thr)
            att_A_ = tf.math.multiply(1 - neighbor_A, bin_mask)
            ss_loss = attention_loss_fn(foreground_A, att_A_)
            
        # SS Loss without constrained background
        else:
            _, background_A = generate_masks(foreground_A, args)
            bin_mask = binarize_mask(att_A, args.ss_attention_thr)
            att_A_ = tf.math.multiply(foreground_A, bin_mask)
            ss_loss = attention_loss_fn(foreground_A, att_A_)

        return foreground_A, background_A, ss_loss

    def train_step(A, B, ep):

        """Trains the generators and the discriminators for one sep.
           Updates the pools with new synthetic images.

        Args:
            A: image with simulation tools blended on a
                surgical background.
            
            B: real surgical image.

        """
        
        # Random flip the tool image, compute the foreground mask and add a random background
        A = random_flip(A)
        foreground_A = tf.image.rgb_to_grayscale(binarize_img(A, 5))
        A = background_addition(A, args)

        # Train generators and attention modules for 1 step
        A2B, B2A = train_G(A, B, foreground_A, ep)

        # Push A2B and B2A in the pool
        A2B = A2B_pool(A2B)
        B2A = B2A_pool(B2A)

        # Train discriminators for 1 step
        train_D(A, B, A2B, B2A, ep)

        return
    
    @tf.function
    def sample(A, B):
        """Predicts the synthetic images over a full cycle.

        Args:
            A: image with simulation tools blended on a
                surgical background.
            
            B: real surgical image.

        Returns:
            A: image with simulation tools blended on a
                surgical background.
            
            B: real surgical image.

            A2B, B2A: synthetic real and simulation images (half cycle), respectively.

            A2B2A, B2A2B: synthetic simulation and real images (full cycle), respectively.

            att_A, att_B: attention maps from attention modules A and B (half cycle), respectively.

            att_A2B, att_B2A: attention maps from attention modules B and A (full cycle), respectively.

        """

        A2B_ = generator_A2B(A)
        B2A_ = generator_B2A(B)

        att_A = attention_A(A)
        att_B = attention_B(B)

        att_A = tf.image.grayscale_to_rgb(att_A)
        att_B = tf.image.grayscale_to_rgb(att_B)


        A2B = tf.math.add(tf.math.multiply(A2B_, att_A), tf.math.multiply(A, 1 - att_A))
        B2A = tf.math.add(tf.math.multiply(B2A_, att_B), tf.math.multiply(B, 1 - att_B))

        A2B2A_ = generator_B2A(A2B)
        B2A2B_ = generator_A2B(B2A)

        att_A2B = attention_B(A2B)
        att_B2A = attention_A(B2A)

        att_A2B = tf.image.grayscale_to_rgb(att_A2B)
        att_B2A = tf.image.grayscale_to_rgb(att_B2A)

        A2B2A = tf.math.add(tf.math.multiply(A2B2A_, att_A2B), tf.math.multiply(A2B, 1 - att_A2B))
        B2A2B = tf.math.add(tf.math.multiply(B2A2B_, att_B2A), tf.math.multiply(B2A, 1 - att_B2A))

        return A, B, A2B, B2A, A2B2A, B2A2B, att_A, att_B, att_A2B, att_B2A


    # ==============================================================================
    # =                                    run                                     =
    # ==============================================================================

    # epoch counter
    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

    # summary
    train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

    # sample
    test_iter = iter(A_B_dataset_test)
    sample_dir = py.join(output_dir, 'samples_training')
    py.mkdir(sample_dir)

    # main loop
    with train_summary_writer.as_default():
        for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):

            # update epoch counter
            ep_cnt.assign_add(1)

            # train for an epoch
            for A, B in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):
                
                train_step(A, B, ep)
                
                # sample
                if G_optimizer_A_att.iterations.numpy() % 2 == 0:
                    A, B = next(test_iter)
                    
                    A = background_addition(A, args)

                    A, B, A2B, B2A, A2B2A, B2A2B, att_A, att_B, att_A2B, att_B2A = sample(A, B)
                    
                    #Save results
                    att_A = 2*att_A - 1
                    att_B = 2*att_B - 1
                    att_A2B = 2*att_A2B - 1
                    att_B2A = 2*att_B2A - 1
                    
                    img = im.immerge(np.concatenate([A, A2B, A2B2A, att_A, att_A2B, B, B2A, B2A2B, att_B, att_B2A], axis=0), n_rows=2)
                    img = cv2.normalize(img, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)                 
                    
                    try:
                        im.imwrite(img, py.join(sample_dir, 'ep_{}_iter-{}.jpg'.format(str(ep).zfill(3), str(G_optimizer_A_att.iterations.numpy()).zfill(6))))
                    except:
                        print('Image has pixel values not in range [-1.0, 1.0]')

                    # Save models
                    if args.start_saving_epoch is not None and ep >= args.start_saving_epoch:
                        model_path = os.path.join(models_path, 'model_{}'.format(str(ep).zfill(3)))
                        create_path(model_path)

                        generator_A2B.save(py.join(model_path, 'A2B_gen_{}'.format(str(ep).zfill(3))), overwrite=True, save_format='tf')
                        attention_A.save(py.join(model_path, 'A_att_{}'.format(str(ep).zfill(3))), overwrite=True, save_format='tf')

                        generator_B2A.save(py.join(model_path, 'B2A_gen_{}'.format(str(ep).zfill(3))), overwrite=True, save_format='tf')
                        attention_B.save(py.join(model_path, 'B_att_{}'.format(str(ep).zfill(3))), overwrite=True, save_format='tf')
        
if __name__ == '__main__':
    py.arg('--datasets_dir', default='/home/ema/my_workspace/datasets/sim2real/MICCAI_2017') # dataset directory
    py.arg('--save_dir', default='/home/ema/my_workspace') # save directory
    py.arg('--backgrounds_dir', default='/home/ema/my_workspace/datasets/backgrounds/MICCAI_2017/backgrounds') # backgrounds directory
    py.arg('--seg_model_path', default='/home/ema/my_workspace/datasets/models/seg_model_2017.hdf5') # path to the pre-training weights for the attention modules.
    py.arg('--load_size', type=int, default=[576, 720])  # load image to this size
    py.arg('--crop_size', type=int, default=512)  # then crop to this size
    py.arg('--batch_size', type=int, default=1) # size of the batch
    py.arg('--epochs', type=int, default=100) # number of epoch
    py.arg('--epoch_decay', type=int, default=75)  # epoch to start decaying learning rate
    py.arg('--lr', type=float, default=0.0002) # learning rate for generators and discriminators
    py.arg('--lr_att', type=float, default=0.0002) # learning rate for the attention modules
    py.arg('--beta_1', type=float, default=0.5) # beta 1 value to setup Adam optimizer
    py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
    py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
    py.arg('--gradient_penalty_weight', type=float, default=0.0) # hyperparam. to weight gradient pealty
    py.arg('--adversarial_loss_weight', type=tuple, default=(1.0, 1.0)) # hyperparam. to weight adversarial loss
    py.arg('--cycle_loss_weight', type=tuple, default=(0.5, 10.0)) # hyperparam. to weight cycle loss
    py.arg('--ssim_loss_weight', type=tuple, default=(0.5, 10.0)) # hyperparam. to weight structure similarity loss
    py.arg('--attention_weight', type=float, default=1.0) # hyperparam. to weight Attention Similarity loss
    py.arg('--sim_supervision_weight', type=float, default=2.0) # hyperparam. to weight Simulation Supervision loss
    py.arg('--ss_attention_thr', type=int, default=25) # threshold to binarize attention maps in SS loss
    py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
    py.arg('--lrD_on_lrG', type=int, default=4) # ratio between discriminator's and generator's learning rates
    py.arg('--start_saving_epoch', type=int, default=-1) # epoch from which to start saving models
    py.arg('--neighbor_kernel_size', type=int, default=25) # size of the neighbor mask (costrained loss only)
    py.arg('--attention_background_constraint', type=bool, default=False) # flag to set the background constraint on the SS loss
    py.arg('--discriminator', type=str, default='ConvDiscriminator', choices=['ConvDiscriminator', 'ConvDiscriminator_no_spectral', 'ConvDiscriminator_no_spectral_25', 'ConvDiscriminator_dropout'])
    args = py.args()

    train(args)
