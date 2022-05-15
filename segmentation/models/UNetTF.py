# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

from keras.layers import *
import tensorflow as tf
import tensorflow.keras as K

from .blocks import *
from utils import *


def UNetTF(input_shape=DEFAULT_TF_INPUT_SHAPE,
           name="UNetTF",
           dropout=0.5,
           kernel_init='he_normal',
           normalize=True,
           up_transpose=True,
           kernel_regularizer=K.regularizers.l2(),
           **kwargs):
    def __build_model(inputs):
        nb_filters = [32,64,128,256,512]
        if up_transpose:
            up_block = Transpose_Block
        else:
            up_block = UpSampleConvo_Block

        down_args = {
            'dropout': dropout,
            'kernel_init':kernel_init,
            'normalize':normalize,
            'kernel_regularizer': kernel_regularizer
        }

        up_args = {
            'dropout': dropout,
            'kernel_init': kernel_init,
            'normalize': normalize,
            'up_convo': up_block,
            'kernel_regularizer': kernel_regularizer
        }

        out_args = {
            'filters': 4,  # number of output classes
            'kernel_size':(1,1),
            'padding':'same',
            'activation':'softmax',
            'kernel_initializer':kernel_init,
            'kernel_regularizer': kernel_regularizer
        }

        convo1,pool1 = Down_Block(name=name+"-down-block-1",filters=nb_filters[0],**down_args)(inputs)
        convo2,pool2 = Down_Block(name=name+"-down-block-2",filters=nb_filters[1],**down_args)(pool1)
        convo3,pool3 = Down_Block(name=name+"-down-block-3",filters=nb_filters[2],**down_args)(pool2)
        convo4,pool4 = Down_Block(name=name+"-down-block-4",filters=nb_filters[3],**down_args)(pool3)

        convo5 = Convo_Block(name=name+"-convo-block",filters=nb_filters[4],**down_args)(pool4)

        up1 = Up_Block(name=name+"-up-block-1",filters=nb_filters[3],**up_args)(x=convo5,merger=[convo4])
        up2 = Up_Block(name=name+"-up-block-2",filters=nb_filters[2],**up_args)(x=up1,   merger=[convo3])
        up3 = Up_Block(name=name+"-up-block-3",filters=nb_filters[1],**up_args)(x=up2,   merger=[convo2])
        up4 = Up_Block(name=name+"-up-block-4",filters=nb_filters[0],**up_args)(x=up3,   merger=[convo1])

        return Conv2D(name=name+"-final-convo",**out_args)(up4)
    
    inputs = K.Input(input_shape)
    outputs = __build_model(inputs)
    model = K.Model(inputs=inputs, outputs=outputs, name=name)
    # store parameters for the Trainer to be able to log them to MLflow
    model.dropout = dropout
    model.kernel_init = kernel_init
    model.normalize = normalize
    model.up_transpose = up_transpose
    model.kernel_regularizer = kernel_regularizer
    return model
