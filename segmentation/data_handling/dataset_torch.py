# code taken and adapted from
# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/ and Pytorch data tutorial

from unittest.mock import DEFAULT
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from data_handling import DataLoader
from utils import *


class CustomDataset(Dataset):
    def __init__(self, data_paths, preprocessing=None, valid_resolutions=None, mode=DEFAULT_MODE):
        self.data_paths = data_paths
        self.preprocessing = preprocessing
        self.valid_resolutions = valid_resolutions
        self.mode = mode

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.data_paths)

    def __getitem__(self, idx):
        json_path = self.data_paths[idx]
        sample_np, gt = DataLoader._create_sample_np(json_path, self.valid_resolutions, self.mode)
        sample_torch_pre = torch.from_numpy(sample_np)
        if self.preprocessing is not None:
            sample_torch = self.preprocessing(sample_torch_pre, is_gt=False)
        else:
            sample_torch = sample_torch_pre

        if self.mode == MODE_SEGMENTATION:
            gt_torch_pre = torch.from_numpy(gt)
            if self.preprocessing is not None:
                gt_torch = self.preprocessing(gt_torch_pre, is_gt=True)
            else:
                gt_torch = gt_torch_pre
            return sample_torch, gt_torch
        else:
            return sample_torch, gt
