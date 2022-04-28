# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

from keras.layers import *
import tensorflow as tf
import tensorflow.keras as K

from .blocks import *
from utils import *


def UNetPlusPlusTF(input_shape=DEFAULT_TF_INPUT_SHAPE,
                   name="UNetPlusPlusTF",
                   dropout=0.5,
                   kernel_init='he_normal',
                   normalize=True,
                   up_transpose=True,
                   average=False,
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

            convo1_1,pool1 = Down_Block(name=name+"-down-block-1",filters=nb_filters[0], **down_args)(inputs)
            convo2_1,pool2 = Down_Block(name=name+"-down-block-2",filters=nb_filters[1], **down_args)(pool1)
            convo3_1,pool3 = Down_Block(name=name+"-down-block-3",filters=nb_filters[2], **down_args)(pool2)
            convo4_1,pool4 = Down_Block(name=name+"-down-block-4",filters=nb_filters[3],**down_args)(pool3)

            convo5_1 = Convo_Block(name=name+"-convo-block-5_1",filters=nb_filters[4],**down_args)(pool4)

            convo1_2 = Up_Block(name=name+"-convo1_2",filters=nb_filters[0],**up_args)(x=convo2_1,merger=[convo1_1])

            convo2_2 = Up_Block(name=name+"-convo2_2",filters=nb_filters[1],**up_args)(x=convo3_1,merger=[convo2_1])
            convo1_3 = Up_Block(name=name+"-convo1_3",filters=nb_filters[0],**up_args)(x=convo2_2,merger=[convo1_1,convo1_2])

            convo3_2 = Up_Block(name=name+"-convo3_2",filters=nb_filters[2],**up_args)(x=convo4_1,merger=[convo3_1])
            convo2_3 = Up_Block(name=name+"-convo2_3",filters=nb_filters[1],**up_args)(x=convo3_2,merger=[convo2_1,convo2_2])
            convo1_4 = Up_Block(name=name+"-convo1_4",filters=nb_filters[0],**up_args)(x=convo2_3,merger=[convo1_1,convo1_2,convo1_3])

            convo4_2 = Up_Block(name=name+"-convo4_2",filters=nb_filters[3],**up_args)(x=convo5_1,merger=[convo4_1])
            convo3_3 = Up_Block(name=name+"-convo3_3",filters=nb_filters[2],**up_args)(x=convo4_2,merger=[convo3_1,convo3_2])
            convo2_4 = Up_Block(name=name+"-convo2_4",filters=nb_filters[1],**up_args)(x=convo3_3,merger=[convo2_1, convo2_2, convo2_3])
            convo1_5 = Up_Block(name=name+"-convo1_5",filters=nb_filters[0],**up_args)(x=convo2_4,merger=[convo1_1, convo1_2, convo1_3, convo1_4])

            output1 = Conv2D(name=name+"-output-1",**out_args)(convo1_2)
            output2 = Conv2D(name=name+"-output-2",**out_args)(convo1_3)
            output3 = Conv2D(name=name+"-output-3",**out_args)(convo1_4)
            output4 = Conv2D(name=name+"-output-4",**out_args)(convo1_5)
            
            if average:
                return Average(name=name+"-final-average")([output1,output2,output3,output4])
            return output4

    inputs = K.Input(input_shape)
    outputs = __build_model(inputs)
    model = K.Model(inputs=inputs, outputs=outputs, name='UNetPlusPlus')
    # store parameters for the Trainer to be able to log them to MLflow
    model.dropout = dropout
    model.kernel_init = kernel_init
    model.normalize = normalize
    model.up_transpose = up_transpose
    model.average = average
    model.kernel_regularizer = kernel_regularizer
    return model
