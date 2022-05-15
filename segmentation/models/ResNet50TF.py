# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

from keras.layers import *
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.initializers import GlorotUniform

import tensorflow.keras.applications.resnet50 
import tensorflow.keras.applications.resnet 

from .blocks import *
from utils import *


def ResNet50(input_shape=DEFAULT_TF_INPUT_SHAPE,
             name="ResNet50TF",
             kernel_init=None, # GlorotUniform(),
             **kwargs):
    # note that we cannot use "include_top=True" or "weights='imagenet'" here, as otherwise the input shape needs to be
    # (224, 224, 3), yet we need 12 channels
    # we also cannot use anything other than pooling=None, since otherwise we get only 2 channels for backbone_features
    # instead of the required 4 
    def __build_model(inputs):
        backbone = tensorflow.keras.applications.resnet50.ResNet50(include_top=False,
                                                                   weights=None,
                                                                   input_tensor=inputs,
                                                                   input_shape=input_shape,
                                                                   classes=3,
                                                                   # kernel_initializer=kernel_init,
                                                                   **kwargs)
        
        if kernel_init is not None:
            for layer in backbone.layers:
                layer_new_weights = []
                all_layer_weights = layer.get_weights()
                if len(all_layer_weights) > 0:
                    for weight_idx, layer_weights in enumerate(all_layer_weights):
                        if 'kernel' in layer.weights[weight_idx].name:
                            weights = kernel_init(layer_weights.shape)
                            layer_new_weights.append(weights)
                        else:
                            layer_new_weights.append(layer_weights)
                    layer.set_weights(layer_new_weights)
        
        backbone_features = backbone(inputs)

        # note that we cannot use a Dense layer if we do not know the shape of the input

        out_args = {
            'filters': 3,  # number of output classes
            'kernel_size': (1, 1),
            'padding':'same',
            'kernel_initializer':kernel_init
        }
        final_convo = Conv2D(name=name+"-final-convo",**out_args)(backbone_features)
        output_pre = GlobalAveragePooling2D()(final_convo)
        output = Activation(activation='softmax')(output_pre)
        return output
        
    inputs = K.Input(input_shape)
    outputs = __build_model(inputs)
    model = K.Model(inputs=inputs, outputs=outputs, name=name)
    # store parameters for the Trainer to be able to log them to MLflow
    
    model.kernel_init = kernel_init

    return model
