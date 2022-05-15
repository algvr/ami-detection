# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

from keras.layers import *
import tensorflow as tf
import tensorflow.keras as K

from .blocks import *
from utils import *


def AttUNetPlusPlusTF(input_shape=DEFAULT_TF_INPUT_SHAPE,
                      name="AttUnetPlusPlusTF-",
                      dropout=0.5,
                      kernel_init='he_normal',
                      normalize=False,
                      deep_supervision=False,
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

        x0_0,pool1 = Down_Block(name=name+"-down-block-1",filters=nb_filters[0],**down_args)(inputs)
        x1_0,pool2 = Down_Block(name=name+"-down-block-2",filters=nb_filters[1],**down_args)(pool1)
        x2_0,pool3 = Down_Block(name=name+"-down-block-3",filters=nb_filters[2],**down_args)(pool2)
        x3_0,pool4 = Down_Block(name=name+"-down-block-4",filters=nb_filters[3],**down_args)(pool3)

        x4_0 = Convo_Block(name=name+"-convo-block-bottom",filters=nb_filters[4],**down_args)(pool4)
        
        att_x00, x0_1 = Attention_PlusPlus_Block(name=name+"-att-pp-1",filters=nb_filters[0],**up_args)(x=x0_0,g=x1_0,down=None,to_concat=[])
        att_x10, x1_1 = Attention_PlusPlus_Block(name=name+"-att-pp-2",filters=nb_filters[1],**up_args)(x=x1_0,g=x2_0,down=x0_1,to_concat=[])
        att_x20, x2_1 = Attention_PlusPlus_Block(name=name+"-att-pp-3",filters=nb_filters[2],**up_args)(x=x2_0,g=x3_0,down=x1_1,to_concat=[])
        att_x30, x3_1 = Attention_PlusPlus_Block(name=name+"-att-pp-4",filters=nb_filters[3],**up_args)(x=x3_0,g=x4_0,down=x2_1,to_concat=[])

        att_x01, x0_2 = Attention_PlusPlus_Block(name=name+"-att-pp-5",filters=nb_filters[0],**up_args)(x=x0_1,g=x1_1,down=None,to_concat=[att_x00])
        att_x11, x1_2 = Attention_PlusPlus_Block(name=name+"-att-pp-6",filters=nb_filters[1],**up_args)(x=x1_1,g=x2_1,down=x0_2,to_concat=[att_x10])
        att_x21, x2_2 = Attention_PlusPlus_Block(name=name+"-att-pp-7",filters=nb_filters[2],**up_args)(x=x2_1,g=x3_1,down=x1_2,to_concat=[att_x20])

        att_x02, x0_3 = Attention_PlusPlus_Block(name=name+"-att-pp-8",filters=nb_filters[0],**up_args)(x=x0_2,g=x1_2,down=None,to_concat=[att_x00,att_x01])
        att_x12, x1_3 = Attention_PlusPlus_Block(name=name+"-att-pp-9",filters=nb_filters[1],**up_args)(x=x1_2,g=x2_2,down=x0_3,to_concat=[att_x10,att_x11])

        att_x03, x0_4 = Attention_PlusPlus_Block(name=name+"-att-pp-x",filters=nb_filters[0],**up_args)(x=x0_3,g=x1_3,down=None,to_concat=[att_x00,att_x01,att_x02])

        output1 = Conv2D(name=name+"-output-1",**out_args)(x0_1)
        output2 = Conv2D(name=name+"-output-2",**out_args)(x0_2)
        output3 = Conv2D(name=name+"-output-3",**out_args)(x0_3)
        output4 = Conv2D(name=name+"-output-4",**out_args)(x0_4)

        if deep_supervision:
            return Average(name=name+"-average")([output1,output2,output3,output4])
        return output4

    inputs = K.Input(input_shape)
    outputs = __build_model(inputs)
    model = K.Model(inputs=inputs, outputs=outputs, name='AttUNetPlusPlus')
    # store parameters for the Trainer to be able to log them to MLflow
    model.dropout = dropout
    model.kernel_init = kernel_init
    model.normalize = normalize
    model.deep_supervision = deep_supervision
    model.up_transpose = up_transpose
    model.kernel_regularizer = kernel_regularizer
    return model
