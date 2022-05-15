# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

import abc

from data_handling import *
from models import *
from trainers import *
from utils import *


class Factory(abc.ABC):
    """Abstract class for the factory method, in order to create corresponding trainer and dataloader for a specific model.
    Use the static method "get_factory(model_name: string) to get the corresponding factory class
    """
    
    @abc.abstractmethod
    def get_trainer_class(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_model_class(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_dataloader_class(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_mode(self):
        raise NotImplementedError 
    
    @staticmethod
    def get_factory(model_name):
        model_name_lower_no_sep = model_name.lower().replace('-', '').replace('_', '')
        if model_name_lower_no_sep in ["unettf", "unettensorflow"]:
            return UNetTFFactory()
        elif model_name_lower_no_sep in ["unet++", "unetplusplus"]:
            return UNetPlusPlusFactory()
        elif model_name_lower_no_sep in ["attunet", "attentionunet"]:
            return AttUNetFactory()
        elif model_name_lower_no_sep in ["attunet++", "attentionunet++", "attentionunetplusplus", "attunetplusplus",
                                         "attunetplusplustf"]:
            return AttUNetPlusPlusTFFactory()
        elif model_name_lower_no_sep in ["resnet50", "resnet50tf"]:
            return ResNet50TFFactory()
        elif model_name_lower_no_sep in ["resnetgenerichead", "resnettfgenerichead",
                                         "resnetgenhead", "resnettfgenhead"]:
            return ResNetTFGenericHeadFactory()
        elif model_name_lower_no_sep in ["resnet152", "resnet152tf"]:
            return ResNet152TFFactory()
        else:
            print(f"The factory for the model {model_name} doesn't exist. Check if you wrote the model name "
                  f"correctly and implemented a corresponding factory in factory.py.")


class UNetTFFactory(Factory):
    def get_trainer_class(self):
        return UNetTFTrainer

    def get_model_class(self):
        return UNetTF

    def get_dataloader_class(self):
        return TFDataLoader

    def get_mode(self):
        return MODE_SEGMENTATION


class UNetPlusPlusFactory(Factory):
    def get_trainer_class(self):
        return UNetPlusPlusTrainer

    def get_model_class(self):
        return UNetPlusPlusTF

    def get_dataloader_class(self):
        return TFDataLoader
        
    def get_mode(self):
        return MODE_SEGMENTATION


class AttUNetFactory(Factory):
    def get_trainer_class(self):
        return AttUNetTrainer

    def get_model_class(self):
        return AttUnetTF

    def get_dataloader_class(self):
        return TFDataLoader
        
    def get_mode(self):
        return MODE_SEGMENTATION


class AttUNetPlusPlusTFFactory(Factory):
    def get_trainer_class(self):
        return AttUNetPlusPlusTrainer

    def get_model_class(self):
        return AttUNetPlusPlusTF

    def get_dataloader_class(self):
        return TFDataLoader
        
    def get_mode(self):
        return MODE_SEGMENTATION


class ResNet50TFFactory(Factory):
    def get_trainer_class(self):
        return ResNet50TFTrainer

    def get_model_class(self):
        return ResNet50TF

    def get_dataloader_class(self):
        return TFDataLoader

    def get_mode(self):
        return MODE_IMAGE_CLASSIFICATION


class ResNetTFGenericHeadFactory(Factory):
    def get_trainer_class(self):
        return ResNet50TFTrainer

    def get_model_class(self):
        return ResNetTFGenericHead

    def get_dataloader_class(self):
        return TFDataLoader

    def get_mode(self):
        return MODE_IMAGE_CLASSIFICATION

class ResNet152TFFactory(Factory):
    def get_trainer_class(self):
        return ResNet50TFTrainer

    def get_model_class(self):
        return ResNet152TF

    def get_dataloader_class(self):
        return TFDataLoader

    def get_mode(self):
        return MODE_IMAGE_CLASSIFICATION
