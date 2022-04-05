import pandas as pd
import numpy as np
import wfdb
import ast


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def get_ecg_array(path='datasets/ptb_xl/', sampling_rate=100):
    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    Y = Y[:100]  # limit amount of data that we load

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)
    return X
