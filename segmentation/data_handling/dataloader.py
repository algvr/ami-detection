# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

import abc
import datetime
import errno
import json
import numpy as np
import os
import warnings
from PIL import Image
import shutil
from typeguard import resolve_forwardref
import urllib3
import zipfile

from datasets.ptb_xl.data_handling import PTB_XL_LEAD_LABELS
from utils import *


LABEL_NORMAL = 0
LABEL_MI = 1
LABEL_NON_MI_ABNORMALITY = 2


class DataLoader(abc.ABC):
    """
    Args:
        dataset (string): type of Dataset
    """

    def __init__(self, dataset, mode=DEFAULT_MODE):
        self.dataset = dataset
        self.mode = mode
        check = self.__download_data(dataset)
        if check == -1:
            raise RuntimeError("Dataset download failed")

        # set data directories
        ds_base = [ROOT_DIR, "datasets", dataset]
        self.training_img_dir = os.path.join(*[*ds_base, "training"])
        self.test_img_dir = os.path.join(*[*ds_base, "test"])

        # note that these lists may either contain strings (self.mode == MODE_IMAGE_CLASSIFICATION),
        # or tuples of strings (self.mode == MODE_SEGMENTATION)
        self.training_data_paths = self.get_data_paths(self.training_img_dir)
        self.test_data_paths = self.get_data_paths(self.test_img_dir)

        # define dataset variables for later usage
        self.training_data = None
        self.testing_data = None
        self.unlabeled_testing_data = None

        self.dataset_hyperparameters = {}
        dataset_hyperparameter_files = ["dataset_hyperparams.json", "dataset_individual_lead_hyperparams.json"]
        for dataset_hyperparameter_fn in dataset_hyperparameter_files:
            file_path = os.path.join(*[*ds_base, dataset_hyperparameter_fn])
            if os.path.exists(file_path):
               with open(file_path) as f:
                    new_hyperparameters = json.loads(f.read())
                    self.dataset_hyperparameters = {**self.dataset_hyperparameters, **new_hyperparameters}

    
    def get_data_paths(self, data_dir):
        data_paths = []
        have_samples = data_dir is not None and os.path.isdir(data_dir)
        if have_samples:
            for filename in os.listdir(data_dir):
                _, ext = os.path.splitext(filename)
                if ext.lower() == '.json':
                    path = os.path.join(data_dir, filename)
                    if self.mode == MODE_IMAGE_CLASSIFICATION:
                        with open(path) as f:
                            try:
                                config = json.loads(f.read())
                                if '_classification_input_img_paths' in config:
                                    discard = False
                                    for _path in config['_classification_input_img_paths']:
                                        if not os.path.exists(_path):
                                            discard = True
                                            break
                                    
                                    if not discard:
                                        data_paths.append(path)
                            except:
                                pass
                    elif self.mode == MODE_SEGMENTATION:
                        with open(path) as f:
                            file_data = json.loads(f.read())
                        data_paths.extend(list(zip(file_data['_lead_img_paths'].values(),
                                                   file_data['_lead_mask_img_paths'].values())))
        return data_paths

    @abc.abstractmethod
    def get_dataset_sizes(self, split):
        """
        Get the sizes of the training, test and unlabeled datasets associated with this DataLoader.
        Args:
            split: training/test splitting ratio \in [0,1]

        Returns:
            Tuple of (int, int, int): sizes of training, test and unlabeled test datasets, respectively,
            in samples
        """
        raise NotImplementedError('must be defined for torch or tensorflow loader')

    @abc.abstractmethod
    def get_training_dataloader(self, split, batch_size, preprocessing=None, valid_resolutions=None, **args):
        """
        Args:
            split (float): training/test splitting ratio \in [0,1]
            batch_size (int): training batch size
            preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                      constructing the native dataloader
            valid_resolutions (list[(int, int)]): list of resolutions in format (width, height) which the model supports; 
                                                  if necessary, the dataloader will automatically select a resolution to
                                                  scale the input samples to so that they can be processed by the model; 
                                                  the selected resolution may not necessarily be optimal w.r.t.
                                                  the minimization of the padding area
                                                  pass None (default) to avoid such an automatic scaling
            **args:for tensorflow e.g.
                img_height (int): training image height in pixels
                img_width (int): training image width in pixels

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError('must be defined for torch or tensorflow loader')

    @abc.abstractmethod
    def get_testing_dataloader(self, batch_size, preprocessing=None, **args):
        """
        Args:
            batch_size (int): training batch size
            preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                      constructing the native dataloader
            **args:for tensorflow e.g.
                img_height (int): training image height in pixels
                img_width (int): training image width in pixels

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError('must be defined for torch or tensorflow loader')

    @abc.abstractmethod
    def get_unlabeled_testing_dataloader(self, batch_size, preprocessing=None, **args):
        """
        Args:
            batch_size (int): training batch size
            preprocessing (function): function taking a raw sample and returning a preprocessed sample to be used when
                                      constructing the native dataloader
            **args: parameters for torch dataloader, e.g. shuffle (boolean)

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError('must be defined for torch or tensorflow loader')

    def get_default_evaluation_interval(self, split, batch_size, num_epochs, num_samples_to_visualize):
        train_dataset_size, test_dataset_size, _ = self.get_dataset_sizes(split)
        iterations_per_epoch = train_dataset_size // batch_size
        # every time EVALUATE_AFTER_PROCESSING_SAMPLES samples are processed, perform an evaluation
        # cap frequency at one evaluation per MIN_EVALUATION_INTERVAL iterations
        EVALUATE_AFTER_PROCESSING_SAMPLES = 200
        MIN_EVALUATION_INTERVAL = 20
        interval = max(MIN_EVALUATION_INTERVAL, EVALUATE_AFTER_PROCESSING_SAMPLES // batch_size)
        return interval

    @abc.abstractmethod
    def load_model(self, path):
        raise NotImplementedError('must be defined for torch or tensorflow loader')

    @abc.abstractmethod
    def save_model(self, model, path):
        raise NotImplementedError('must be defined for torch or tensorflow loader')

    def __download_data(self, dataset_name):
        destination_path = os.path.join(*[ROOT_DIR, "datasets", dataset_name.lower()])
        ts_path = os.path.join(destination_path, "download_timestamp.txt")
        zip_path = f"{destination_path}.zip"

        url = next((v for k, v in DATASET_ZIP_URLS.items() if dataset_name.lower() == k.lower()), None)
        if url is None:
            warnings.warn(f"Dataset '{dataset_name}' unknown... error in Dataloader.__download_data()")
            return -1

        # check if data already downloaded; use timestamp file written *after* successful download for the check
        if os.path.exists(ts_path):
            return 1
        else:
            os.makedirs(destination_path, exist_ok=True)

        # data doesn't exist yet
        print("Downloading dataset...")
        pool = urllib3.PoolManager()
        try:
            with pool.request("GET", url, preload_content=False) as response, open(zip_path, "wb") as file:
                shutil.copyfileobj(response, file)
        except Exception as e:
            warnings.warn(f"Error encountered while downloading dataset '{dataset_name}': {str(e)}")
            return -1
        print("...done!")

        print("Extracting ZIP archive...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(destination_path)
            print("...done!")
        print("Removing ZIP file...")
        os.unlink(zip_path)
        print("...done!")

        with open(ts_path, "w") as file:
            file.write(str(datetime.datetime.now()))

        return 1

    @staticmethod
    def _get_valid_sample_resolution(original_width, original_height, valid_resolutions):
        # we want to avoid downscaling if possible
        # if it is possible to upscale, we want to add as little padding as possible
        min_gain_area_upscale = np.inf
        min_gain_area_upscale_width, min_gain_area_upscale_height = None, None

        # if it's not possible to upscale, we want the downscaled image to have a maximally high resolution, to keep
        # all details (we don't care about minimizing padding: there could e.g. be a valid resolution with the same
        # aspect ratio, yet with a very small area that causes the curve to become uninterpretable)
        # note that "used area" does not mean there is actually a lead that uses all this area, as the max. width
        # and max. height may come from two distinct leads
        max_used_area_downscale = 0.0
        max_used_area_downscale_width, max_used_area_downscale_height = None, None

        for res in valid_resolutions:
            res_width, res_height = res
            if original_width <= res_width and original_height <= res_height:
                gain_area_upscale = res_width * res_height - original_width * original_height
                if gain_area_upscale < min_gain_area_upscale:
                    min_gain_area_upscale = gain_area_upscale
                    min_gain_area_upscale_width = res_width
                    min_gain_area_upscale_height = res_height
                    if gain_area_upscale == 0:
                        break  # perfect fit
            else:
                width_height_ratio = original_width / original_height
                new_width = res_width
                new_height = new_width / width_height_ratio
                if new_height > res_height:
                    new_height = res_height
                    new_width = new_height * width_height_ratio
                
                used_area_downscale = new_width * new_height
                if used_area_downscale > max_used_area_downscale:
                    max_used_area_downscale = used_area_downscale
                    max_used_area_downscale_width = res_width
                    max_used_area_downscale_height = res_height

        if None not in [min_gain_area_upscale_width, min_gain_area_upscale_height]:
            final_width = min_gain_area_upscale_width
            final_height = min_gain_area_upscale_height
        elif None not in [max_used_area_downscale_width, max_used_area_downscale_width]:
            final_width = max_used_area_downscale_width
            final_height = max_used_area_downscale_height
        else:
            raise RuntimeError('This should not have happened.')
        
        return final_width, final_height

    @staticmethod
    def _create_sample_np(path, valid_resolutions, mode):
        if mode == MODE_SEGMENTATION:
            sample_img = Image.open(path[0]).convert('RGB')
            gt_img = Image.open(path[1]).convert('RGB')

            original_width, original_height = sample_img.size
            final_width, final_height = DataLoader._get_valid_sample_resolution(original_width, original_height,
                                                                                valid_resolutions)
            
            if original_width == final_width and original_height == final_height:
                return np.array(sample_img), np.array(gt_img)
            
            sample_output = Image.new('RGB', (final_width, final_height), (0, 0, 0))
            gt_output = Image.new('RGB', (final_width, final_height), (0, 0, 0))
            sample_output.paste(sample_img, (0, 0))  # paste to top-left corner
            gt_output.paste(gt_img, (0, 0))  # paste to top-left corner

            return np.array(sample_output), np.array(gt_output)
        elif mode == MODE_IMAGE_CLASSIFICATION:
            with open(path, 'r') as f:
                data = json.loads(f.read())
            #if '_lead_img_paths' not in data:
            #    raise RuntimeError('Please extract individual lead images from the ECG dataset using '
            #                        'data_augmentation.ipynb.')
            if '_classification_input_img_paths' not in data:
                raise RuntimeError('Please extract input images for classification from the ECG dataset using '
                                    'request_processing_main.py.')
            if '_is_normal' not in data or '_is_mi' not in data:
                raise RuntimeError('Please ensure the samples are labelled using the boolean _is_normal and _is_mi '
                                   'properties in the respective JSON files.')

            if data['_is_normal']:
                gt = 0
            elif data['_is_mi']:
                gt = 1
            else:  # non-MI abnormality
                gt = 2

            # construct sample

            channel_img_list = []
            with Image.open(data['_merged_classification_input_img_path']).convert('L') as img:
                for channel_idx in range(3):
                    channel_img_list.append(255 - np.array(img))
            sample = np.stack(channel_img_list, axis=0)
            sample = np.moveaxis(sample, 0, -1)  # convert to HWC format
            return sample, gt

            ##########

            # channel_img_list = []
            # for _classification_input_img_path in data['_classification_input_img_paths']:
            #     with Image.open(_classification_input_img_path) as img:
            #         channel_img_list.append(np.array(img))

            # sample = np.stack(channel_img_list)
            # sample = np.moveaxis(sample, 0, -1)  # convert to HWC format
            # return sample, gt

            ##########

            # lead_img_paths = data['_lead_img_paths']
            # lead_imgs = {}
            # largest_width = 0
            # largest_height = 0
            # for lead_name, path in lead_img_paths.items():
            #     # perform further augmentations (contrast increase, etc.) here
            #     lead_img = Image.open(path).convert('L')  # convert to grayscale
            #     lead_imgs[lead_name] = lead_img
            #     lead_img_width, lead_img_height = lead_img.size
            #     largest_width = max(largest_width, lead_img_width)
            #     largest_height = max(largest_height, lead_img_height)

            # if valid_resolutions is not None:
            #     final_width, final_height = DataLoader._get_valid_sample_resolution(largest_width, largest_height)
            # else:
            #     final_width, final_height = largest_width, largest_height

            # np_channel_list = []
            # # now, we need to ensure an order that is consistent across recordings
            # # we use the PTB-XL order here
            # for channel_idx, ptbxl_lead_name in enumerate(PTB_XL_LEAD_LABELS):
            #     channel_img = Image.new('L', (final_width, final_height), 255)  # initialize with all white
            #     # look for matching lead in current recording's leads
            #     for lead_name, lead_img in lead_img_paths.items():
            #         # do not perform a substring matching (due to e.g. VL and aVL being different leads)
            #         if lead_name.lower() == ptbxl_lead_name.lower():
            #             lead_img_width, lead_img_height = lead_imgs[lead_name].size
            #             if lead_img_width <= final_width and lead_img_height <= final_height:
            #                 channel_img.paste(lead_imgs[lead_name], (0, 0))  # paste to top-left corner
            #             else:
            #                 shrinked = lead_imgs[lead_name].thumbnail((final_width, final_height))
            #                 channel_img.paste(shrinked, (0, 0))  # paste to top-left corner
            #             np_channel_list.append(np.array(channel_img))
            #             break
            # sample = np.stack(np_channel_list)
            # sample = np.moveaxis(sample, 0, -1)  # convert to HWC format
            # return sample, gt
        else:
            raise NotImplementedError()

    def get_dataset_hyperparameters(self):
        return self.dataset_hyperparameters
