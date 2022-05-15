# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

"""
Global constants and helper functions
"""

import math
import os
import pathlib
import random
from stat import S_ISDIR, S_ISREG
import sys
import time


###########################################################################################
##################################    global constants    #################################
###########################################################################################

MODE_SEGMENTATION = 0
MODE_IMAGE_CLASSIFICATION = 1
MODE_TIME_SERIES_CLASSIFICATION = 2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # go two dirs up from the dir this file is in
SESSION_ID = int(time.time() * 1000)  # import time of utils.py in milliseconds will be the session ID
IS_DEBUG = getattr(sys, 'gettrace', None) is not None and getattr(sys, 'gettrace', lambda: None)() is not None
ACCEPTED_IMAGE_EXTENSIONS = [".png", ".jpeg", ".jpg", ".gif"]
DEFAULT_SEGMENTATION_THRESHOLD = 0.5
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_MODE = MODE_IMAGE_CLASSIFICATION
DEFAULT_NUM_SAMPLES_TO_VISUALIZE = 36
DATASET_ZIP_URLS = {
    "backgrounds": "https://polybox.ethz.ch/index.php/s/7hE6WIct12CZi66/download",
    "ptb_xl": "https://polybox.ethz.ch/index.php/s/6cYYSheXDP6ZiC5/download",
    "ptb_v": "TODO: REPLACE PLACEHOLDER WITH PROPER URL ONCE DATASET CURATED",
    "ptb_v_080522": "https://polybox.ethz.ch/index.php/s/ti3QbYllkalkDOn/download",
    "ptb_v_classification": "https://polybox.ethz.ch/index.php/s/nPfMjAtDprflZTP/download"
}
# in case multiple jobs are running in the same directory, SESSION_ID will prevent name conflicts
CODEBASE_SNAPSHOT_ZIP_NAME = f"codebase_{SESSION_ID}.zip"
CHECKPOINTS_DIR = os.path.join("checkpoints", str(SESSION_ID))
LOGGING_DIR = "logs/"
MLFLOW_USER = "mlflow_user"
MLFLOW_HOST = "algvrithm.com"
MLFLOW_TRACKING_URI = f"http://{MLFLOW_HOST}:8008"
MLFLOW_JUMP_HOST = "eu-login-01"
MLFLOW_PASS_URL = "https://algvrithm.com/files/mlflow_dstses_pass.txt"
MLFLOW_PROFILING = False
# Pushbullet access token to use for sending notifications about critical events such as exceptions during training
# (None to avoid sending Pushbullet notifications)
DEFAULT_PUSHBULLET_ACCESS_TOKEN = pathlib.Path('pb_token.txt').read_text() if os.path.isfile('pb_token.txt') else None


CLASSIFICATION_RNN_TIMESTEPS_PER_SECOND = 100
LOGGING_IMG_WIDTH = 1000
LOGGING_IMG_HEIGHT = 4000

# gives input size of 480x480 per channel
CLASSIFICATION_NETWORK_MAX_LEAD_WIDTH = 480
CLASSIFICATION_NETWORK_LEAD_HEIGHT = 120  # height for a single lead

DEFAULT_TF_INPUT_SHAPE = ({MODE_SEGMENTATION:         (None, None, 3),
                           MODE_IMAGE_CLASSIFICATION: (CLASSIFICATION_NETWORK_MAX_LEAD_WIDTH,
                                                       4 * CLASSIFICATION_NETWORK_LEAD_HEIGHT, 3)})[DEFAULT_MODE]

###########################################################################################
##################################    helper functions    #################################
###########################################################################################

def consistent_shuffling(*args):
    """
    Randomly permutes all lists in the input arguments such that elements at the same index in all lists are still
    at the same index after the permutation.
    """
    z = list(zip(*args))
    random.shuffle(z)
    return list(map(list, zip(*z)))


def next_perfect_square(n):
    next_n = math.floor(math.sqrt(n)) + 1
    return next_n * next_n


def is_perfect_square(n):
    x = math.sqrt(n)
    return (x - math.floor(x)) == 0


# Cross-platform SFTP directory downloading code adapted from https://stackoverflow.com/a/50130813
def sftp_download_dir_portable(sftp, remote_dir, local_dir, preserve_mtime=False):
    # sftp: pysftp connection object
    for entry in sftp.listdir_attr(remote_dir):
        remote_path = remote_dir + "/" + entry.filename
        local_path = os.path.join(local_dir, entry.filename)
        mode = entry.st_mode
        if S_ISDIR(mode):
            try:
                os.mkdir(local_path)
            except OSError:     
                pass
            sftp_download_dir_portable(sftp, remote_path, local_path, preserve_mtime)
        elif S_ISREG(mode):
            sftp.get(remote_path, local_path, preserve_mtime=preserve_mtime)
 
