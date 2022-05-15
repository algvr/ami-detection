# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

from keras.layers import *
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.initializers import GlorotUniform

import tensorflow.keras.applications.resnet50 
import tensorflow.keras.applications.resnet 

from .blocks import *
from utils import *


def ResNet50TF(input_shape=DEFAULT_TF_INPUT_SHAPE,
               name="ResNet50TF",
               kernel_init=None, # GlorotUniform(),
               num_final_fc_units=None,
               **kwargs):
    # note that we cannot use "include_top=True" or "weights='imagenet'" here, as otherwise the input shape needs to be
    # (224, 224, 3), yet we need 12 channels
    # we also cannot use anything other than pooling=None, since otherwise we get only 2 channels for backbone_features
    # instead of the required 4 
    def __build_model(inputs):
        backbone = tensorflow.keras.applications.resnet50.ResNet50(include_top=False,
                                                                   weights='imagenet',
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
            'padding': 'same',
            'kernel_initializer': kernel_init
        }
        final_convo = Conv2D(name=name+"-final-convo",**out_args)(backbone_features)
        output_pre = GlobalAveragePooling2D()(final_convo)
        if num_final_fc_units not in [0, None]:
            output_pre = Dense(units=num_final_fc_units, activation=None, kernel_initializer=kernel_init)(output_pre)
        output = Activation(activation='softmax')(output_pre)
        return output
        
    inputs = K.Input(input_shape)
    outputs = __build_model(inputs)
    model = K.Model(inputs=inputs, outputs=outputs, name=name)
    # store parameters for the Trainer to be able to log them to MLflow
    
    model.kernel_init = kernel_init
    model.num_final_fc_units = num_final_fc_units

    return model


def ResNetTFGenericHead(input_shape=DEFAULT_TF_INPUT_SHAPE,
                        name=None,
                        kernel_init=None, # GlorotUniform(),
                        backbone_model="ResNet50",
                        train_backbone=True,
                        head_structure=[512, 256, 128, 64],
                        use_imagenet_weights=True,
                        dropout=0.0,
                        **kwargs):
    if name is None:
        name = backbone_model + "TFGenericHead"
    
    # note that we cannot use "include_top=True" or "weights='imagenet'" here, as otherwise the input shape needs to be
    # (224, 224, 3), yet we need 12 channels
    # we also cannot use anything other than pooling=None, since otherwise we get only 2 channels for backbone_features
    # instead of the required 4 
    def __build_model(inputs):
        backbone_class = {"resnet152": tensorflow.keras.applications.resnet.ResNet152,
                          "resnet101": tensorflow.keras.applications.resnet.ResNet101,
                          "resnet50": tensorflow.keras.applications.resnet.ResNet50}[backbone_model.lower().strip()]
        backbone = backbone_class(include_top=False,
                                  weights='imagenet' if use_imagenet_weights else None,
                                  input_tensor=inputs,
                                  input_shape=input_shape,
                                  classes=3,
                                  # kernel_initializer=kernel_init,
                                  **kwargs)

        # if kernel_init is not None:
        #     for layer in backbone.layers:
        #         layer_new_weights = []
        #         all_layer_weights = layer.get_weights()
        #         if len(all_layer_weights) > 0:
        #             for weight_idx, layer_weights in enumerate(all_layer_weights):
        #                 if 'kernel' in layer.weights[weight_idx].name:
        #                     weights = kernel_init(layer_weights.shape)
        #                     layer_new_weights.append(weights)
        #                 else:
        #                     layer_new_weights.append(layer_weights)
        #             layer.set_weights(layer_new_weights)
        

        for layer in backbone.layers:
            backbone.trainable = train_backbone

        backbone_features = backbone(inputs)

        # note that we cannot use a Dense layer if we do not know the shape of the input

        out_args = {
            'filters': 3,  # number of output classes
            'kernel_size': (1, 1),
            'padding': 'same',
            'kernel_initializer': kernel_init
        }
        
        final_convo = Conv2D(name=name+"-final-convo",**out_args)(backbone_features)

        gap_pre = GlobalAveragePooling2D()(final_convo)

        fc_pre = Flatten()(gap_pre)

        if head_structure is not None and len(head_structure) > 0:
            for num_units in head_structure:
                fc_pre = Dense(num_units, activation='relu', kernel_initializer=kernel_init)(fc_pre)
                if dropout > 0.0:
                    fc_pre = Dropout(rate=dropout)(fc_pre)

        final_pre = Dense(3, activation=None, kernel_initializer=kernel_init)(fc_pre)  # final layer
        output = Activation(activation='softmax')(final_pre)
        return output
        
    inputs = K.Input(input_shape)
    outputs = __build_model(inputs)
    model = K.Model(inputs=inputs, outputs=outputs, name=name)
    # store parameters for the Trainer to be able to log them to MLflow
    
    model.kernel_init = kernel_init
    model.head_structure = head_structure
    model.dropout = dropout
    model.backbone_model = backbone_model
    model.train_backbone = train_backbone
    model.use_imagenet_weights = use_imagenet_weights

    return model



def ResNet152TF(input_shape=DEFAULT_TF_INPUT_SHAPE,
                name="ResNet152TF",
                kernel_init=None, # GlorotUniform(),
                num_final_fc_units=None,
                use_imagenet_weights=True,
                **kwargs):
    # note that we cannot use "include_top=True" or "weights='imagenet'" here, as otherwise the input shape needs to be
    # (224, 224, 3), yet we need 12 channels
    # we also cannot use anything other than pooling=None, since otherwise we get only 2 channels for backbone_features
    # instead of the required 4
    def __build_model(inputs):
        backbone = tensorflow.keras.applications.resnet.ResNet152(include_top=False,
                                                                  weights='imagenet' if use_imagenet_weights else None,
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
            'padding': 'same',
            'kernel_initializer': kernel_init
        }
        final_convo = Conv2D(name=name+"-final-convo", **out_args)(backbone_features)
        output_pre = GlobalAveragePooling2D()(final_convo)
        if num_final_fc_units not in [0, None]:
            output_pre = Dense(units=num_final_fc_units, activation=None, kernel_initializer=kernel_init)(output_pre)
        output = Activation(activation='softmax')(output_pre)

        # TODO: add additional blocks!

        return output
        
    inputs = K.Input(input_shape)
    outputs = __build_model(inputs)
    model = K.Model(inputs=inputs, outputs=outputs, name=name)
    # store parameters for the Trainer to be able to log them to MLflow

    model.kernel_init = kernel_init
    model.num_final_fc_units = num_final_fc_units

    return model

