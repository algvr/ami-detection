import pandas as pd
import numpy as np
import wfdb
import ast


# default order of leads in PTB-XL dataset
# (last dimension of ECG recordings returned by "get_ecg_array" and "load_raw_data")
PTB_XL_LEAD_LABELS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def load_raw_data(df, sampling_rate, path):
    """
    Internal function to load and return raw ECG recordings from the PTB-XL dataset; use "get_ecg_array" instead
    :param df: Pandas DataFrame with metadata of ECG recordings
    :param sampling_rate: 100 to use low-resolution recordings; otherwise, high-resolution recordings will be used
    :param path: path to root directory of PTB-XL dataset
    :return: numpy array of dimension [patient, timestep, lead];
             units are mV; lead layout is given in PTB_XL_LEAD_LABELS
    """
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def get_ecg_array(path='datasets/ptb_xl/', sampling_rate=100, max_samples=100):
    """
    Load and return raw ECG recordings and associated metadata from the PTB-XL dataset
    :param path: path to root directory of PTB-XL dataset
    :param sampling_rate: 100 to use low-resolution recordings; otherwise, high-resolution recordings will be used
    :param max_samples: maximum number of samples to load, or None to load all samples
    :return: tuple (X, Y), where X is numpy array of shape [patient, timestep, lead], and Y is metadata of associated
             ECG recordings with the same columns as in "ptbxl_database.csv"
    """
    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    if max_samples is not None:
        Y = Y[:max_samples]  # limit amount of data that we load

    # load raw signal data
    X = load_raw_data(Y, sampling_rate, path)
    return X, Y


def lead_idx_to_lead_label(lead_idx):
    """
    Return the label of the lead at the given index (this is specific to PTB-XL)
    :param lead_idx: index of the lead
    :return: label of associated lead, or None if the given index does not correspond to a valid lead
    """
    lbl = next((lbl for idx, lbl in enumerate(PTB_XL_LEAD_LABELS) if idx == lead_idx), None)
    return lbl


def layout_array_to_label_array(layout_array):
    """
    Traverse the given layout array and construct a matching label array with the name of the lead at each
    corresponding position of the label array
    :param layout_array: layout array of dimension [rows, columns, 3]
    :return: label array of dimension [rows, columns]
    """
    rows, columns, _ = layout_array.shape
    ret_array = []
    for row_idx in range(rows):
        row_array = []
        for column_idx in range(columns):
            lead_idx = layout_array[row_idx, column_idx, 0]
            row_array.append(lead_idx_to_lead_label(lead_idx))
        ret_array.append(row_array)
    return ret_array
