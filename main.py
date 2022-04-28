# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

"""Main runner file. Looks for the "--model" or "-m" command line argument to determine the model to use,
then passes the remaining command line arguments to the constructor of the corresponding model's class."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import itertools
import json
import os
import re

from factory import Factory
from utils import *
from utils.logging import pushbullet_logger


def main():
    # all args that cannot be matched to the Trainer or DataLoader classes and are not in filter_args will be passed to the
    # model's constructor

    trainer_args = ['experiment_name', 'E', 'run_name', 'R', 'split', 's', 'num_epochs', 'e', 'batch_size', 'b',
                    'optimizer_or_lr', 'l', 'loss_function', 'L', 'loss_function_hyperparams', 'H',
                    'evaluation_interval', 'i', 'num_samples_to_visualize', 'v', 'checkpoint_interval', 'c',
                    'load_checkpoint_path', 'C', 'segmentation_threshold', 't']
    dataloader_args = ['dataset', 'd']

    # list of other arguments to avoid passing to constructor of model class
    filter_args = ['h', 'model', 'm']

    parser = argparse.ArgumentParser(description='Implementation of ETHZ CIL Road Segmentation 2022 project')
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-E', '--experiment_name', type=str, required=True)
    parser.add_argument('-R', '--run_name', type=str, required=False)
    parser.add_argument('-s', '--split', type=float, default=DEFAULT_TRAIN_FRACTION, required=False)
    parser.add_argument('-e', '--num_epochs', type=int, required=False)
    parser.add_argument('-b', '--batch_size', type=int, required=False)
    parser.add_argument('-l', '--optimizer_or_lr', type=float, required=False)
    parser.add_argument('-L', '--loss_function', type=str, required=False)
    # json.loads: substitute for dict
    parser.add_argument('-H', '--loss_function_hyperparams', type=json.loads, required=False)
    parser.add_argument('-i', '--evaluation_interval', type=float, required=False)
    parser.add_argument('-v', '--num_samples_to_visualize', type=int, required=False)
    parser.add_argument('-c', '--checkpoint_interval', type=int, required=False)
    parser.add_argument('-C', '--load_checkpoint_path', type=str, required=False)
    parser.add_argument('-t', '--segmentation_threshold', type=float, default=DEFAULT_SEGMENTATION_THRESHOLD, required=False)
    parser.add_argument('-d', '--dataset', type=str, required=True)
    known_args, unknown_args = parser.parse_known_args()

    remove_leading_dashes = lambda s: ''.join(itertools.dropwhile(lambda c: c == '-', s))
    # float check taken from https://thispointer.com/check-if-a-string-is-a-number-or-float-in-python/
    cast_arg = lambda s: s[1:-1] if s.startswith('"') and s.endswith('"')\
                         else int(s) if remove_leading_dashes(s).isdigit()\
                         else float(s) if re.search('[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$', s) is not None\
                         else s.lower() == 'true' if s.lower() in ['true', 'false']\
                         else None if s.lower() == 'none'\
                         else eval(s) if any([s.startswith('(') and s.endswith(')'),
                                              s.startswith('[') and s.endswith(']'),
                                              s.startswith('{') and s.endswith('}')])\
                         else s

    known_args_dict = dict(map(lambda arg: (arg, getattr(known_args, arg)), vars(known_args)))
    unknown_args_dict = dict(map(lambda arg: (remove_leading_dashes(arg.split('=')[0]),
                                            cast_arg([*arg.split('='), True][1])),
                                unknown_args))
    arg_dict = {**known_args_dict, **unknown_args_dict}

    factory = Factory.get_factory(known_args.model)
    dataloader = factory.get_dataloader_class()(**{k: v for k, v in arg_dict.items() if k.lower() in dataloader_args})
    model = factory.get_model_class()(**{k: v for k, v in arg_dict.items() if k.lower() not in [*trainer_args,
                                                                                                *dataloader_args,
                                                                                                *filter_args]})
    trainer = factory.get_trainer_class()(dataloader=dataloader, model=model,
                                        **{k: v for k, v in arg_dict.items() if k.lower() in trainer_args})

    # do not move these Pushbullet messages into the Trainer class, as this may lead to a large amount of
    # messages when using Hyperopt

    if not IS_DEBUG:
        pushbullet_logger.send_pushbullet_message('Training started.\n' +\
                                                  f'Hyperparameters:\n{trainer._get_hyperparams()}')

    last_test_loss = trainer.train()
    
    if not IS_DEBUG:
        pushbullet_logger.send_pushbullet_message(('Training finished. Last test loss: %.4f\n' % last_test_loss) +\
                                                  f'Hyperparameters:\n{trainer._get_hyperparams()}')


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
        main()
    else:
        try:
            print(f'Session ID: {SESSION_ID}\n'
                'Not running in debug mode\n'
                'stderr and stdout will be written to "%s" and "%s", respectively\n' % (stderr_path, stdout_path))
            # buffering=1: use line-by-line buffering
            with open(stderr_path, 'w', buffering=1) as stderr_f, open(stdout_path, 'w', buffering=1) as stdout_f:
                with redirect_stderr(stderr_f), redirect_stdout(stdout_f):
                    main()
        except Exception as e:
            err_msg = f'*** Exception encountered: ***\n{e}'
            pushbullet_logger.send_pushbullet_message(err_msg)
            raise e
