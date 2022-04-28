# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

import abc
import inspect
import json
from losses import *
import mlflow
import numpy as np
import os
import pexpect
import paramiko
import pysftp
import requests
import shutil
import socket
import tensorflow.keras as K
import time

from data_handling import DataLoader
from utils import *
from utils.logging import mlflow_logger, optim_hyparam_serializer


class Trainer(abc.ABC):
    def __init__(self, dataloader, model, experiment_name=None, run_name=None, split=None, num_epochs=None,
                 batch_size=None, optimizer_or_lr=None, loss_function=None, loss_function_hyperparams=None,
                 evaluation_interval=None, num_samples_to_visualize=None, checkpoint_interval=None,
                 load_checkpoint_path=None, segmentation_threshold=None):
        """
        Abstract class for model trainers.
        Args:
            dataloader: the DataLoader to use when training the model
            model: the model to train
            experiment_name: name of the experiment to log this training run under in MLflow
            run_name: name of the run to log this training run under in MLflow (None to use default name assigned by
                      MLflow)
            split: fraction of dataset provided by the DataLoader which to use for training rather than test
                   (None to use default)
            num_epochs: number of epochs, i.e. passes through the dataset, to train model for (None to use default)
            batch_size: number of samples to use per training iteration (None to use default)
            optimizer_or_lr: optimizer to use, or learning rate to use with this method's default optimizer
                             (None to use default)
            loss_function: (name of) loss function to use (None to use default)
            loss_function_hyperparams: hyperparameters of loss function to use
                                       (will be bound to the loss function automatically; None to skip)
            evaluation_interval: interval, in iterations, in which to perform an evaluation on the test set
                                 (None to use default)
            num_samples_to_visualize: ignore this argument
            checkpoint_interval: interval, in iterations, in which to create model checkpoints
                                 specify an extremely high number (e.g. 1e15) to only create a single checkpoint after training has finished
                                 (WARNING: None or 0 to discard model)
            load_checkpoint_path: path to checkpoint file, or SFTP checkpoint URL for MLflow, to load a checkpoint and
                                  resume training from (None to start training from scratch instead)
            segmentation_threshold: ignore this argument
        """
        self.dataloader = dataloader
        self.model = model
        self.mlflow_experiment_name = experiment_name
        self.mlflow_experiment_id = None
        self.mlflow_run_name = run_name
        self.split = split
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer_or_lr = optimizer_or_lr
        self.loss_function_hyperparams = loss_function_hyperparams if loss_function_hyperparams is not None else {}

        self.loss_function_name = str(loss_function)
        if isinstance(loss_function, str):
            # additional imports to expand scope of losses accessible via "eval"
            import torch
            import torch.nn
            import tensorflow.keras.losses
            self.loss_function = eval(loss_function)
        else:
            self.loss_function = loss_function
        if inspect.isclass(self.loss_function):
            # instantiate class with given hyperparameters
            self.loss_function = self.loss_function(**self.loss_function_hyperparams)
        elif inspect.isfunction(self.loss_function):
            self.orig_loss_function = self.loss_function
            self.loss_function = lambda *args, **kwargs: self.orig_loss_function(*args, **kwargs,
                                                                                 **self.loss_function_hyperparams)

        self.evaluation_interval = evaluation_interval
        self.num_samples_to_visualize =\
            num_samples_to_visualize if num_samples_to_visualize is not None else DEFAULT_NUM_SAMPLES_TO_VISUALIZE
        self.checkpoint_interval = checkpoint_interval
        self.do_checkpoint = self.checkpoint_interval is not None and self.checkpoint_interval > 0
        self.segmentation_threshold =\
            segmentation_threshold if segmentation_threshold is not None else DEFAULT_SEGMENTATION_THRESHOLD
        self.is_windows = os.name == 'nt'
        self.load_checkpoint_path = load_checkpoint_path
        if not self.do_checkpoint:
            print('\n*** WARNING: no checkpoints of this model will be created! Specify valid checkpoint_interval '
                  '(in iterations) to Trainer in order to create checkpoints. ***\n')

        self.valid_resolutions = self._load_valid_resolutions()

    def _init_mlflow(self):
        self.mlflow_experiment_id = None
        if self.mlflow_experiment_name is not None:
            def add_known_hosts(host, user, password, jump_host=None):
                spawn_str =\
                    'ssh %s@%s' % (user, host) if jump_host is None else 'ssh -J %s %s@%s' % (jump_host, user, host)
                if self.is_windows:
                    # pexpect.spawn not supported on windows
                    import wexpect
                    child = wexpect.spawn(spawn_str)
                else:
                    child = pexpect.spawn(spawn_str)
                i = child.expect(['.*ssword.*', '.*(yes/no).*'])
                if i == 1:
                    child.sendline('yes')
                    child.expect('.*ssword.*')
                child.sendline(password)
                child.expect('.*')
                time.sleep(1)
                child.sendline('exit')

                if jump_host is not None:
                    if self.is_windows:
                        raise RuntimeError('Use of jump hosts for Trainer not supported on Windows machines')

                    # monkey-patch pysftp to use the provided jump host

                    def new_start_transport(self, host, port):
                        try:
                            jumpbox = paramiko.SSHClient()
                            jumpbox.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                            jumpbox.connect(jump_host,
                                            key_filename=os.path.join(*[os.getenv('HOME'), '.ssh', 'id_' + jump_host]))

                            jumpbox_transport = jumpbox.get_transport()
                            dest_addr = (host, port)
                            jumpbox_channel = jumpbox_transport.open_channel('direct-tcpip', dest_addr, ('', 0))

                            target = paramiko.SSHClient()
                            target.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                            target.connect(host, port, user, password, sock=jumpbox_channel)

                            self._transport = target.get_transport()
                            self._transport.connect = lambda *args, **kwargs: None  # ignore subsequent "connect" calls

                            # set security ciphers if set
                            if self._cnopts.ciphers is not None:
                                ciphers = self._cnopts.ciphers
                                self._transport.get_security_options().ciphers = ciphers
                        except (AttributeError, socket.gaierror):
                            # couldn't connect
                            raise pysftp.ConnectionException(host, port)

                    pysftp.Connection._start_transport = new_start_transport

            mlflow_init_successful = True
            MLFLOW_INIT_ERROR_MSG = 'MLflow initialization failed. Will not use MLflow for this run.'

            try:
                mlflow_pass = requests.get(MLFLOW_PASS_URL).text
                try:
                    add_known_hosts(MLFLOW_HOST, MLFLOW_USER, mlflow_pass)
                except:
                    add_known_hosts(MLFLOW_HOST, MLFLOW_USER, mlflow_pass, MLFLOW_JUMP_HOST)
            except:
                mlflow_init_successful = False
                print(MLFLOW_INIT_ERROR_MSG)

            if mlflow_init_successful:
                try:
                    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                    experiment = mlflow.get_experiment_by_name(self.mlflow_experiment_name)
                    if experiment is None:
                        self.mlflow_experiment_id = mlflow.create_experiment(self.mlflow_experiment_name)
                    else:
                        self.mlflow_experiment_id = experiment.experiment_id
                except:
                    mlflow_init_successful = False
                    print(MLFLOW_INIT_ERROR_MSG)

            return mlflow_init_successful
        else:
            return False

    def _get_hyperparams(self):
        """
        Returns a dict of what is considered a hyperparameter
        Please add any hyperparameter that you want to be logged to MLFlow
        """
        return {
            'split': self.split,
            'epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'loss_function': self.loss_function_name if hasattr(self, 'loss_function_name') else self.loss_function,
            # 'seg_threshold': self.segmentation_threshold,
            'model': self.model.name if hasattr(self.model, 'name') else type(self.model).__name__,
            'dataset': self.dataloader.dataset,
            'from_checkpoint': self.load_checkpoint_path if self.load_checkpoint_path is not None else '',
            'session_id': SESSION_ID,
            **({f'dataset_{k}': v for k, v in self.dataloader.get_dataset_hyperparameters().items()}),
            **(optim_hyparam_serializer.serialize_optimizer_hyperparams(self.optimizer_or_lr)),
            **({f'loss_{k}': v for k, v in self.loss_function_hyperparams.items()})
        }

    def _load_valid_resolutions(self):
        """
        Returns a (non-exhaustive) list of input resolutions supported by this model, in the format (width, height).
        """
        model_name = self.model.name if hasattr(self.model, 'name') else type(self.model).__name__
        json_file_path = os.path.join('valid_res_lists', f'{model_name}.json')
        if not os.path.exists(json_file_path):
            raise RuntimeError(f'Valid resolution list for model "{model_name}" not found '
                               f'(looked for "{json_file_path}").')
        with open(json_file_path, 'r') as f:
            config_data = json.loads(f.read())
            return config_data['valid_resolutions']

    @staticmethod
    @abc.abstractmethod
    def get_default_optimizer_with_lr(lr, model):
        """
        Constructs and returns the default optimizer for this method, with the given learning rate and model.
        Args:
            lr: the learning rate to use
            model: the model to use

        Returns: optimizer object (subclass-dependent)
        """
        raise NotImplementedError('Must be defined for trainer.')

    @abc.abstractmethod
    def _fit_model(self, mlflow_run):
        """
        Fit the model.
        """
        raise NotImplementedError('Must be defined for trainer.')

    def train(self):
        """
        Trains the model
        """
        if self.do_checkpoint and not os.path.exists(CHECKPOINTS_DIR):
            os.makedirs(CHECKPOINTS_DIR)
        
        if self.mlflow_experiment_name is not None and self._init_mlflow():
            with mlflow.start_run(experiment_id=self.mlflow_experiment_id, run_name=self.mlflow_run_name) as run:
                try:
                    mlflow_logger.log_hyperparams(self._get_hyperparams())
                    mlflow_logger.snapshot_codebase()  # snapshot before training as the files may change in-between
                    mlflow_logger.log_codebase()  # log codebase before training, to be invariant to training crashes and stops
                    last_test_loss = self._fit_model(mlflow_run=run)
                    if self.do_checkpoint:
                        mlflow_logger.log_checkpoints()
                    mlflow_logger.log_logfiles()
                except Exception as e:
                    err_msg = f'*** Exception encountered: ***\n{e}'
                    print(f'\n\n{err_msg}\n')
                    mlflow_logger.log_logfiles()
                    if not IS_DEBUG:
                        pushbullet_logger.send_pushbullet_message(err_msg)
                    raise e
        else:
            last_test_loss = self._fit_model(mlflow_run=None)

        if os.path.exists(CHECKPOINTS_DIR):
            shutil.rmtree(CHECKPOINTS_DIR)

        return last_test_loss

    @staticmethod
    def _fill_images_array(preds, batch_ys, images):
        if batch_ys is None:
            batch_ys = np.zeros_like(preds)
        if len(batch_ys.shape) > len(preds.shape):
            # collapse channel dimension
            batch_ys = np.argmax(batch_ys, axis=-1)

        # color scheme: correctly predicted ECG curve: green
        #               wrongly predicted ECG curve: red
        #               missed ECG curve (made whatever prediction): blue
        #               correctly predicted line: dark green for vertical, darker green for horizontal
        #               wrongly predicted line: dark red for vertical, darker red for horizontal
        #               (no missed lines)
        #               same priority order as in mask generation
        


        correct_ecg_curve = np.logical_and(preds == 3, batch_ys == 3).astype(np.float32)
        wrong_ecg_curve = np.logical_and(preds == 3, ~(batch_ys == 3)).astype(np.float32)
        missed_ecg_curve = np.logical_and(~(preds == 3), batch_ys == 3).astype(np.float32)

        correct_hor_line = np.logical_and(preds == 1, batch_ys == 1).astype(np.float32)
        wrong_hor_line = np.logical_and(preds == 1, ~(batch_ys == 1)).astype(np.float32)

        correct_vert_line = np.logical_and(preds == 2, batch_ys == 2).astype(np.float32)
        wrong_vert_line = np.logical_and(preds == 2, ~(batch_ys == 2)).astype(np.float32)

        correct_ecg_curve_color =\
            np.concatenate((np.full_like(correct_ecg_curve, 0.0),
                            np.full_like(correct_ecg_curve, 255.0 / 255.0),
                            np.full_like(correct_ecg_curve, 0.0)), axis=1)
                                                  
        wrong_ecg_curve_color =\
            np.concatenate((np.full_like(correct_ecg_curve, 255.0 / 255.0),
                            np.full_like(correct_ecg_curve, 0.0 / 255.0),
                            np.full_like(correct_ecg_curve, 0.0 / 255.0)), axis=1)

        missed_ecg_curve_color =\
            np.concatenate((np.full_like(correct_ecg_curve, 0.0 / 255.0),
                            np.full_like(correct_ecg_curve, 0.0 / 255.0),
                            np.full_like(correct_ecg_curve, 255.0 / 255.0)), axis=1)
                            
        correct_vert_line_color =\
            np.concatenate((np.full_like(correct_ecg_curve, 0.0 / 255.0),
                            np.full_like(correct_ecg_curve, 170.0 / 255.0),
                            np.full_like(correct_ecg_curve, 0.0 / 255.0)), axis=1)
                            
        wrong_vert_line_color =\
            np.concatenate((np.full_like(correct_ecg_curve, 170.0 / 255.0),
                            np.full_like(correct_ecg_curve, 0.0 / 255.0),
                            np.full_like(correct_ecg_curve, 0.0 / 255.0)), axis=1)

        correct_hor_line_color =\
            np.concatenate((np.full_like(correct_ecg_curve, 0.0 / 255.0),
                            np.full_like(correct_ecg_curve, 85.0 / 255.0),
                            np.full_like(correct_ecg_curve, 0.0 / 255.0)), axis=1)

        wrong_hor_line_color =\
            np.concatenate((np.full_like(correct_ecg_curve, 85.0 / 255.0),
                            np.full_like(correct_ecg_curve, 0.0 / 255.0),
                            np.full_like(correct_ecg_curve, 0.0 / 255.0)), axis=1)


        #curve_mask_arr = (np.array(mask_imgs[0])[:, :, 3] > 0).astype(np.uint8)
        #thick_hor_lines_mask_arr = (np.array(mask_imgs[1])[:, :, 3] > 0).astype(np.uint8)
        #thick_vert_lines_mask_arr = (np.array(mask_imgs[2])[:, :, 3] > 0).astype(np.uint8)

        #mask_arr = ((thick_hor_lines_mask_arr * 85 * (1 - thick_vert_lines_mask_arr) + thick_vert_lines_mask_arr * 170) * 
        #            (1 - curve_mask_arr)) + curve_mask_arr * 255
        rgb = ((((correct_hor_line * correct_hor_line_color + wrong_hor_line * wrong_hor_line_color)
                * (1 - (correct_vert_line + wrong_vert_line)))
                + (correct_vert_line * correct_vert_line_color + wrong_vert_line * wrong_vert_line_color))
               * (1 - (correct_ecg_curve + wrong_ecg_curve + missed_ecg_curve))
               + (correct_ecg_curve * correct_ecg_curve_color + wrong_ecg_curve * wrong_ecg_curve_color
                  + missed_ecg_curve * missed_ecg_curve_color))

        for batch_sample_idx in range(preds.shape[0]):
            images.append(rgb[batch_sample_idx])

    @staticmethod
    def _save_image_array(images, file_path):

        def segmentation_to_image(x):
            x = (x * 255).astype(int)
            if len(x.shape) < 3:  # if this is true, there are probably bigger problems somewhere else
                x = np.expand_dims(x, axis=0)  # CHW format
            return x

        n = len(images)
        if is_perfect_square(n):
            nb_cols = math.sqrt(n)
        else:
            nb_cols = math.sqrt(next_perfect_square(n))
        nb_cols = int(nb_cols)  # Need it to be an integer
        nb_rows = math.ceil(float(n) / float(nb_cols))  # Number of rows in final image

        # Append enough black images to complete the last non-empty row
        while len(images) < nb_cols * nb_rows:
            images.append(np.zeros_like(images[0]))
        arr = []  # Store images concatenated in the last dimension here
        # First, determine width of widest row, and height of highest column
        
        widest_row_width = 0
        highest_col_height = 0

        # Images are in CHW format
        for i in range(nb_rows):
            row_width = sum([img.shape[2] for img in images[(i * nb_cols):(i + 1) * nb_cols]])
            if row_width > widest_row_width:
                widest_row_width = row_width
        
                
        for i in range(nb_rows):
            # make sure all images in this row have equal height; make sure the row is "widest_row_width" wide by
            # appending a padding image to the right

            max_col_height = max([images[i * nb_cols + j].shape[-2] for j in range(nb_cols)])

            row_imgs = []
            aggregated_row_width = 0
            for img in images[(i * nb_cols):(i + 1) * nb_cols]:
                height, width = img.shape[-2:]
                pad_arr = [*([(0, 0)] * len(img.shape[:-2])),
                            (0, max_col_height - height),
                            (0, 0)]
                row_imgs.append(np.pad(img, pad_arr, 'constant'))
                aggregated_row_width += width
            if aggregated_row_width < widest_row_width:
                row_imgs.append(np.zeros((*images[0].shape[:-2], max_col_height, widest_row_width - aggregated_row_width),
                                         dtype=images[0].dtype))
            
            row = np.concatenate(row_imgs, axis=-1)
            arr.append(row)
        # Concatenate in the second-to-last dimension to get the final big image
        final = np.concatenate(arr, axis=-2)
        K.preprocessing.image.save_img(file_path, segmentation_to_image(final), data_format="channels_first")
