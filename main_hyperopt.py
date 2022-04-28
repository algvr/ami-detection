# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

"""Hyperopt runner file. Looks for the "--search_space" or "-s" command line argument to determine the feature_space to use,
then passes the remaining command line arguments to a Hyperparameter optimizer."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from hyperopt_.HyperOptimizer import HyperParamOptimizer
from hyperopt_ import *
import os
import time
import warnings

from utils import *
from utils.logging import pushbullet_logger


def main_hyperopt():
    warnings.filterwarnings("ignore", category=UserWarning) 
    # list of supported arguments
    filter_args = ['h', 'search_space', 's', 'num_runs', 'n']

    parser = argparse.ArgumentParser(description='Implementation of Hyperparameter Optimizer that searches through the parameter space')
    parser.add_argument('-s', '--search_space', type=str, help="Give the name of the search space, which has to be defined in 'hyperopt\\param_space_...' and imported in main_hyperopt")
    parser.add_argument('-n', '--num_runs', type = int, help="The number of Hyperopt runs, aka models trained with different parameters from the search space")
    params = parser.parse_args()

    try:
        optimizer = HyperParamOptimizer(eval(params.search_space), params.search_space)
        optimizer.run(n_runs=params.num_runs)
    except NameError as n:
        print(f"Check if you wrote the search space name correctly. Error message:\n{n}")


if __name__ == '__main__':
    abs_logging_dir = os.path.join(ROOT_DIR, LOGGING_DIR)
    if not os.path.isdir(abs_logging_dir):
        os.makedirs(abs_logging_dir)

    stderr_path = os.path.join(abs_logging_dir, f'stderr_{SESSION_ID}.log')
    stdout_path = os.path.join(abs_logging_dir, f'stdout_{SESSION_ID}.log')

    for path in [stderr_path, stdout_path]:
        if os.path.isfile(path):
            os.unlink(path)
    
    if IS_DEBUG:
        main_hyperopt()
    else:
        try:
            print(f'Session ID: {SESSION_ID}\n'
                'Not running in debug mode\n'
                'stderr and stdout will be written to "%s" and "%s", respectively\n' % (stderr_path, stdout_path))
            # buffering=1: use line-by-line buffering
            with open(stderr_path, 'w', buffering=1) as stderr_f, open(stdout_path, 'w', buffering=1) as stdout_f:
                with redirect_stderr(stderr_f), redirect_stdout(stdout_f):
                    main_hyperopt()
        except Exception as e:
            err_msg = f'*** Exception encountered: ***\n{e}'
            pushbullet_logger.send_pushbullet_message(err_msg)
            raise e
