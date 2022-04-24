"""
Global constants and helper functions
"""

import os
import random
import math


###########################################################################################
##################################    global constants    #################################
###########################################################################################

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_FRACTION = 0.8
DATASET_ZIP_URLS = {
    "backgrounds": "https://polybox.ethz.ch/index.php/s/7hE6WIct12CZi66/download",
    "ptb_xl": "https://polybox.ethz.ch/index.php/s/6cYYSheXDP6ZiC5/download"
}
CODEBASE_SNAPSHOT_ZIP_NAME = "codebase_snapshot.zip"
CHECKPOINTS_DIR = "checkpoints/"
MLFLOW_USER = "mlflow_user"
MLFLOW_HOST = "algvrithm.com"
MLFLOW_TRACKING_URI = f"http://{MLFLOW_HOST}:8008"
MLFLOW_JUMP_HOST = "eu-login-01"
MLFLOW_PASS_URL = "https://algvrithm.com/files/mlflow_dstses_pass.txt"
MLFLOW_PROFILING = True


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
