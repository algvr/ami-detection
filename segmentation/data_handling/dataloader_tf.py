# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

import itertools
import json
import numpy as np
from PIL import Image
import random
import tensorflow as tf
import warnings

from .dataloader import DataLoader
from utils import *


class TFDataLoader(DataLoader):
    def __init__(self, dataset, mode=DEFAULT_MODE):
        super().__init__(dataset, mode)

    # Get the sizes of the training, test and unlabeled datasets associated with this DataLoader.
    # Args:
    #   split   (float): training/test splitting ratio \in [0,1]
    # Returns: Tuple of (int, int, int): sizes of training, test and unlabeled test datasets, respectively,
    #          in samples
    def get_dataset_sizes(self, split):
        dataset_size = len(self.training_data_paths)
        train_size = int(dataset_size * split)
        test_size = dataset_size - train_size
        unlabeled_test_size = len(self.test_data_paths)
        return train_size, test_size, unlabeled_test_size


    # Create a single sample
    # Args:
    #    preprocessing
    #    json_path  (string): path to a JSON file with information about lead images belonging to a single recording
    #    channels   (int): 4 for RGBA, 3 for RGB, 1 for grayscale, 0 - default encoding
    #    valid_resolutions (list[(int, int)]): list of resolutions in format (width, height) which the model supports; 
    #                                          if necessary, the dataloader will automatically select a resolution to
    #                                          scale the input samples to so that they can be processed by the model; 
    #                                          the selected resolution may not necessarily be optimal w.r.t.
    #                                          the minimization of the padding area
    #                                          pass None (default) to avoid such an automatic scaling
    # Returns: tuple: (Tensor of type dtype uint8, label)
    ###  
    def __create_sample(self, json_path, preprocessing=None, channels=0, valid_resolutions=None):
        sample_np, gt = super()._create_sample_np(json_path, valid_resolutions, self.mode)
        sample_tf_pre = tf.convert_to_tensor(sample_np)
        if preprocessing is not None:
            sample_tf = preprocessing(sample_tf_pre, is_gt=False)
        else:
            sample_tf = sample_tf_pre

        if self.mode == MODE_SEGMENTATION:
            gt_tf_pre = tf.convert_to_tensor(gt)
            if preprocessing is not None:
                gt_tf = preprocessing(gt_tf_pre, is_gt=True)
            else:
                gt_tf = gt_tf_pre
            return sample_tf, gt_tf
        else:
            return sample_tf, tf.constant(gt, dtype=tf.dtypes.uint8)

    # Get images
    # Args:
    #    image_dir  (string): the directory of images
    # Returns: Dataset
    ###  
    def __get_image_data(self, data_paths, shuffle=True, preprocessing=None, offset=0, length=1e12,
                         valid_resolutions=None):
        # WARNING: must use lambda captures (see https://stackoverflow.com/q/10452770)
        data_paths_trimmed = data_paths[offset:offset+length]
        parse_sample = lambda x, preprocessing=preprocessing: self.__create_sample(x, preprocessing,
                                                                                   valid_resolutions=valid_resolutions)

        # itertools.count() gives infinite generators

        test_sample = parse_sample(data_paths_trimmed[0])
        output_types = (test_sample[0].dtype, test_sample[1].dtype)
        if shuffle:
            data_paths_shuffled = [*data_paths_trimmed]
            random.shuffle(data_paths_shuffled)
            return tf.data.Dataset.from_generator(
                lambda: (parse_sample(data_path) for _ in itertools.count() for data_path in data_paths_shuffled),
                output_types=output_types)
        else:
            return tf.data.Dataset.from_generator(
                lambda: (parse_sample(data_path) for _ in itertools.count() for data_path in data_paths_trimmed),
                output_types=output_types)

    ###
    # Create training/validation dataset split
    # Args:
    #    split (float): training/test splitting ratio, e.g. 0.8 for 80"%" training and 20"%" test data
    #    batch_size (int): training batch size
    #    preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
    #                              constructing the native dataloader
    #    valid_resolutions (list[(int, int)]): list of resolutions in format (width, height) which the model supports; 
    #                                          if necessary, the dataloader will automatically select a resolution to
    #                                          scale the input samples to so that they can be processed by the model; 
    #                                          the selected resolution may not necessarily be optimal w.r.t.
    #                                          the minimization of the padding area
    #                                          pass None (default) to avoid such an automatic scaling
    # Returns: Dataset
    ###  
    def get_training_dataloader(self, split, batch_size, preprocessing=None, valid_resolutions=None, **args):
        # Get images' names and data

        # WARNING: the train/test splitting behavior must be consistent across TFDataLoader and TorchDataLoader,
        # and may not be stochastic, so as to ensure comparability across models/runs
        # for the same reason, while the training set should be shuffled, the test set should not

        dataset_size = len(self.training_data_paths)
        train_size = int(dataset_size * split)
        test_size = dataset_size - train_size

        self.training_data = self.__get_image_data(self.training_data_paths, shuffle=True, preprocessing=preprocessing,
                                                   offset=0, length=train_size,
                                                   valid_resolutions=valid_resolutions)
        self.testing_data = self.__get_image_data(self.training_data_paths, shuffle=False, preprocessing=preprocessing,
                                                  offset=train_size, length=test_size,
                                                  valid_resolutions=valid_resolutions)
        print(f'Train data consists of ({train_size}) samples')
        print(f'Test data consists of ({test_size}) samples')

        return self.training_data.batch(batch_size)
    
    ###
    # Get labeled dataset for validation
    # Args:
    #    batch_size (int): training batch size
    #    preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
    #                              constructing the native dataloader
    # Returns: Dataset
    ###       
    def get_testing_dataloader(self, batch_size, preprocessing=None, **args):
        if self.testing_data is None:
            warnings.warn("You called test dataloader before training dataloader. "
                          "Usually the test data is created by splitting the training data when calling "
                          "get_training_dataloader. If groundtruth test data is explicitly available in the dataset, "
                          "this will be used, otherwise the complete training dataset will be used. "
                          "Call <get_unlabeled_testing_dataloader()> in order to get the test data of a dataset "
                          "without annotations.")
            self.testing_data = self.__get_image_data(self.training_data_paths, shuffle=False,
                                                        preprocessing=preprocessing)
            print(f'Test data consists of ({len(self.training_data_paths)}) samples')
        return self.testing_data.batch(batch_size)

    ###
    # Get unlabeled dataset for test
    # Args:
    #    batch_size (int): training batch size
    #    preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
    #                              constructing the native dataloader
    # Returns: Dataset
    ###       
    def get_unlabeled_testing_dataloader(self, batch_size, preprocessing=None, **args):
        if self.unlabeled_testing_data is None:
            self.unlabeled_testing_data = self.__get_image_data(self.test_data_paths, preprocessing=preprocessing,
                                                                shuffle=False)
            print(f'Found ({len(self.test_data_paths)}) unlabeled test samples')
        return self.unlabeled_testing_data.batch(batch_size)
        
    # Load model
    # Args:
    #   filepath (string)
    def load_model(self, filepath):
        return tf.keras.models.load_model(filepath)
        
    # Save model
    # Args:
    #   model (Keras.model)
    #   filepath (string)
    def save_model(self, model, filepath):
        tf.keras.models.save_model(model, filepath)
