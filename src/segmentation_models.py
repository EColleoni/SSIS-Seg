import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization
from spectral import SpectralConv2D, SpectralConv2DTranspose
from attention import SelfAttnModel
import tensorflow.keras as keras
import numpy as np
import segmentation_models as sm
sm.set_framework('tf.keras')



def segmentation_model(input_sh=(512, 512, 3)):
    model = sm.Unet('vgg16', input_shape=input_sh, encoder_weights=None, classes=1, activation='sigmoid')

    return model

def model_Unet_sim2real(input_shape=(512, 512, 3)):
    model = sm.Unet('resnet34', input_shape=input_shape, classes=1, activation='sigmoid', encoder_weights='imagenet')
    return model

def model_Unet_sim2real_multiclass():
    model = sm.Unet('resnet34', classes=3, activation='sigmoid', encoder_weights='imagenet')
    return model

def segmentation_resnet18(input_sh=(512, 512, 3)):
    model = sm.Unet('resnet18', input_shape=input_sh, encoder_weights='imagenet', classes=1, activation='sigmoid')

    return model    
    
if __name__ == '__main__':
    
    model = segmentation_model((512, 512, 3))
