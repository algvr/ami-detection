# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

import warnings
import torch
from torch.utils.data import DataLoader as torch_dl, Subset
from .dataset_torch import CustomDataset
from .dataloader import DataLoader
import utils
from models import *


class TorchDataLoader(DataLoader):
    def __init__(self, dataset):
        super().__init__(dataset)

    def get_dataset_sizes(self, split):
        """
        Get the sizes of the training, test and unlabeled datasets associated with this DataLoader.
        Args:
            split: training/test splitting ratio \in [0,1]

        Returns:
            Tuple of (int, int, int): sizes of training, test and unlabeled test datasets, respectively,
            in samples
        """
        full_training_data_len = len(self.training_data_paths)
        training_data_len = int(full_training_data_len * split)
        testing_data_len = full_training_data_len - training_data_len
        unlabeled_testing_data_len = len(self.test_data_paths)
        return training_data_len, testing_data_len, unlabeled_testing_data_len

    def get_training_dataloader(self, split, batch_size, preprocessing=None, valid_resolutions=None, **args):
        """
        Args:
            split (float): training/test splitting ratio, e.g. 0.8 for 80"%" training and 20"%" test data
            batch_size (int): training batch size
            preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                      constructing the native dataloader
            **args: parameters for torch dataloader, e.g. shuffle (boolean)
            valid_resolutions (list[(int, int)]): list of resolutions in format (width, height) which the model supports; 
                                                  if necessary, the dataloader will automatically select a resolution to
                                                  scale the input samples to so that they can be processed by the model; 
                                                  the selected resolution may not necessarily be optimal w.r.t.
                                                  the minimization of the padding area
                                                  pass None (default) to avoid such an automatic scaling
        Returns:
            Torch Dataloader
        """
        #load training data and possibly split
        
        # WARNING: the train/test splitting behavior must be consistent across TFDataLoader and TorchDataLoader,
        # and may not be stochastic, so as to ensure comparability across models/runs
        # for the same reason, while the training set should be shuffled, the test set should not

        dataset = CustomDataset(self.training_data_paths, preprocessing, valid_resolutions)
        training_data_len = int(len(dataset)*split)
        testing_data_len = len(dataset)-training_data_len
        
        self.training_data = Subset(dataset, list(range(training_data_len)))
        self.testing_data = Subset(dataset, list(range(training_data_len, len(dataset))))
        
        return torch_dl(self.training_data, batch_size, shuffle=True, **args)
    
    def get_testing_dataloader(self, batch_size, preprocessing=None, **args):
        """
        Args:
            batch_size (int): training batch size
            preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                      constructing the native dataloader
            **args: parameters for torch dataloader, e.g. shuffle (boolean)

        Returns:
            Torch Dataloader
        """
        # WARNING: the train/test splitting behavior must be consistent across TFDataLoader and TorchDataLoader,
        # and may not be stochastic, so as to ensure comparability across models/runs
        # for the same reason, while the training set should be shuffled, the test set should not

        if self.testing_data is None:
            warnings.warn("You called test dataloader before training dataloader. "
                          "Usually the test data is created by splitting the training data when calling "
                          "get_training_dataloader. If groundtruth test data is explicitly available in the dataset, "
                          "this will be used, otherwise the complete training dataset will be used. "
                          "Call <get_unlabeled_testing_dataloader()> in order to get the test data of a dataset "
                          "without annotations.")
            self.testing_data = CustomDataset(self.training_data_paths, preprocessing)

        return torch_dl(self.testing_data, batch_size, shuffle=False, **args)
            
    def get_unlabeled_testing_dataloader(self, batch_size, preprocessing=None, **args):
        """
        Args:
            batch_size (int): training batch size
            preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                      constructing the native dataloader
            **args: parameters for torch dataloader, e.g. shuffle (boolean)

        Returns:
            Torch Dataloader
        """
        if self.unlabeled_testing_data is None:
            self.unlabeled_testing_data = CustomDataset(*utils.consistent_shuffling(self.test_data_paths), preprocessing)
        return torch_dl(self.unlabeled_testing_data, batch_size, shuffle=False, **args)

    def load_model(self, path, model_class_as_string):
        model = eval(model_class_as_string)
        model.load_state_dict(torch.load(path))
        return model
    
    def save_model(self, model, path):
        torch.save(model.state_dict(), path)
