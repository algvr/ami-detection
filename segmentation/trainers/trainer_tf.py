# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

import abc
import datetime
import keras
import keras.backend
import numpy as np
import os
import pysftp
import requests
import shutil
import tempfile
import tensorflow as tf
import tensorflow.keras.callbacks as KC
from urllib.parse import urlparse

from losses.loss_harmonizer import DEFAULT_TF_DIM_LAYOUT
from losses.precision_recall_f1 import *
from utils.logging import mlflow_logger
from .trainer import Trainer
from utils import *


class TFTrainer(Trainer, abc.ABC):
    def __init__(self, dataloader, model, preprocessing, steps_per_training_epoch,
                 experiment_name=None, run_name=None, split=None, num_epochs=None, batch_size=None,
                 optimizer_or_lr=None, loss_function=None, loss_function_hyperparams=None, evaluation_interval=None,
                 num_samples_to_visualize=None, checkpoint_interval=None, load_checkpoint_path=None,
                 segmentation_threshold=None):
        """
        Abstract class for TensorFlow-based model trainers.
        Args:
            dataloader: the DataLoader to use when training the model
            model: the model to train
        """
        super().__init__(dataloader, model, experiment_name, run_name, split, num_epochs, batch_size, optimizer_or_lr,
                         loss_function, loss_function_hyperparams, evaluation_interval, num_samples_to_visualize,
                         checkpoint_interval, load_checkpoint_path, segmentation_threshold)
        # these attributes must also be set by each TFTrainer subclass upon initialization:
        self.preprocessing = preprocessing
        self.steps_per_training_epoch = steps_per_training_epoch

    # Subclassing tensorflow.keras.callbacks.Callback (here: KC.Callback) allows us to override various functions to be
    # called when specific events occur while fitting a model using TF's model.fit(...). An instance of the subclass
    # needs to be passed in the "callbacks" parameter (which, if specified, can either be a single instance, or a list
    # of instances of KC.Callback subclasses)
    class Callback(KC.Callback):
        def __init__(self, trainer, mlflow_run):
            super().__init__()
            self.trainer = trainer
            self.mlflow_run = mlflow_run
            self.do_evaluate = self.trainer.evaluation_interval is not None and self.trainer.evaluation_interval > 0
            self.iteration_idx = 0
            self.epoch_iteration_idx = 0
            self.epoch_idx = 0
            self.do_visualize = self.trainer.num_samples_to_visualize is not None and \
                                self.trainer.num_samples_to_visualize > 0

        def on_train_begin(self, logs=None):
            print('\nTraining started at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
            print(f'Session ID: {SESSION_ID}')
            print('Hyperparameters:')
            print(self.trainer._get_hyperparams())
            print('')

        def on_train_end(self, logs=None):
            print('\nTraining finished at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
            mlflow_logger.log_logfiles()

        def on_epoch_begin(self, epoch, logs=None):
            pass

        def on_epoch_end(self, epoch, logs=None):
            print('\n\nEpoch %i finished at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) % epoch)
            print('Metrics: %s\n' % str(logs))
            mlflow_logger.log_metrics(logs, aggregate_iteration_idx=self.iteration_idx)
            mlflow_logger.log_logfiles()
            
            # checkpoints should be logged to MLflow right after their creation, so that if training is
            # stopped/crashes *without* reaching the final "mlflow_logger.log_checkpoints()" call in trainer.py,
            # prior checkpoints have already been persisted
            # since we don't have a way of getting notified when KC.ModelCheckpoint has finished creating the checkpoint,
            # we simply check at the end of each epoch whether there are any checkpoints to upload and upload them
            # if necessary
            mlflow_logger.log_checkpoints()
            
            self.epoch_idx += 1
            self.epoch_iteration_idx = 0

            # it seems we can only safely delete the original checkpoint dir after having trained for at least one
            # iteration
            if os.path.isdir(f'original_checkpoint_{SESSION_ID}.ckpt'):
                shutil.rmtree(f'original_checkpoint_{SESSION_ID}.ckpt')


        def on_train_batch_begin(self, batch, logs=None):
            pass

        def on_train_batch_end(self, batch, logs=None):
            if self.do_evaluate and self.iteration_idx % self.trainer.evaluation_interval == 0:
                metrics = {}
                for classes in ([[3], [1, 2], [1, 2, 3]] if self.trainer.dataloader.mode == MODE_SEGMENTATION\
                                else [[0], [1], [2], [1, 2], [0, 1, 2]]):
                    precision, recall, f1_score = self.trainer.get_precision_recall_F1_score_validation(classes)
                    class_str = "_".join(map(str, classes))
                    metrics = {**metrics, f'precision__{class_str}': precision,
                                          f'recall__{class_str}': recall,
                                          f'f1_score__{class_str}': f1_score}
                print('\nMetrics at aggregate iteration %i (ep. %i, ep.-it. %i): %s'
                      % (self.iteration_idx, self.epoch_idx, batch, str(metrics)))
                if mlflow_logger.logging_to_mlflow_enabled():
                    mlflow_logger.log_metrics(metrics, aggregate_iteration_idx=self.iteration_idx)
                    if self.do_visualize and self.trainer.dataloader.mode == MODE_SEGMENTATION:
                        mlflow_logger.log_visualizations(self.trainer, self.iteration_idx)
            
            if self.trainer.do_checkpoint\
                and self.iteration_idx % self.trainer.checkpoint_interval == 0\
                and self.iteration_idx > 0:  # avoid creating checkpoints at iteration 0
                checkpoint_path = f'{CHECKPOINTS_DIR}/cp_ep-{"%05i" % self.epoch_idx}'+\
                                  f'_it-{"%05i" % self.epoch_iteration_idx}' +\
                                  f'_step-{self.iteration_idx}.ckpt'
                keras.models.save_model(model=self.trainer.model, filepath=checkpoint_path)
                mlflow_logger.log_checkpoints()
            
            self.iteration_idx += 1
            self.epoch_iteration_idx += 1

    # Visualizations are created using mlflow_logger's "log_visualizations" (containing ML framework-independent code),
    # and the "create_visualizations" functions of the Trainer subclasses (containing ML framework-specific code)
    # Specifically, the Trainer calls mlflow_logger's "log_visualizations" (e.g. in "on_train_batch_end" of the
    # tensorflow.keras.callbacks.Callback subclass), which in turn uses the Trainer's "create_visualizations".
    def create_visualizations(self, file_path):
        # for batch_xs, batch_ys in self.test_loader.shuffle(10 * num_samples).batch(num_samples):

        # fix half of the samples, randomize other half
        # the first, fixed half of samples serves for comparison purposes across models/runs
        # the second, randomized half allows us to spot weaknesses of the model we might miss when
        # always visualizing the same samples

        num_to_visualize = self.num_samples_to_visualize
        # never exceed the given training batch size, else we might face memory problems
        vis_batch_size = min(num_to_visualize, self.batch_size)

        _, test_dataset_size, _ = self.dataloader.get_dataset_sizes(split=self.split)
        if num_to_visualize >= test_dataset_size:
            # just visualize the entire test set
            vis_dataloader = self.test_loader.take(test_dataset_size).batch(vis_batch_size)
        else:
            num_fixed_samples = num_to_visualize // 2
            num_random_samples = num_to_visualize - num_fixed_samples

            fixed_samples = self.test_loader.take(num_fixed_samples)
            random_samples = self.test_loader.skip(num_fixed_samples).take(num_random_samples).shuffle(num_random_samples)
            vis_dataloader = fixed_samples.concatenate(random_samples).batch(vis_batch_size)

        images = []

        for (batch_xs, batch_ys) in vis_dataloader:
            batch_xs = tf.squeeze(batch_xs, axis=1)
            batch_ys = tf.squeeze(batch_ys, axis=1).numpy()
            if len(batch_ys.shape) < 4:
                batch_ys = np.expand_dims(batch_ys, axis=-1)
            output = self.model.predict(batch_xs)  # returns np.ndarray
            channel_dim_idx = DEFAULT_TF_DIM_LAYOUT.find('C')
            if output.shape[channel_dim_idx] > 1:
                output = np.argmax(output, axis=channel_dim_idx)
                output = np.expand_dims(output, axis=channel_dim_idx)

            preds = output.astype(np.float)
            # preds = np.expand_dims(preds, axis=1)  # so add it back, in CHW format
            batch_ys = np.moveaxis(batch_ys, -1, 1)  # TODO only do this if we know the network uses HWC format
            preds = np.moveaxis(preds, -1, 1)
            # At this point we should have preds.shape = (batch_size, 1, H, W) and same for batch_ys
            self._fill_images_array(preds, batch_ys, images)

        self._save_image_array(images, file_path)

    def _compile_model(self):
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer_or_lr)

    def _load_checkpoint(self, checkpoint_path):
        print(f'\n*** WARNING: resuming training from checkpoint "{checkpoint_path}" ***\n')
        load_from_sftp = checkpoint_path.lower().startswith('sftp://')
        if load_from_sftp:
            # in TF, even though the checkpoint names all end in ".ckpt", they are actually directories
            # hence we have to use sftp_download_dir_portable to download them
            final_checkpoint_path = f'original_checkpoint_{SESSION_ID}.ckpt'
            os.makedirs(final_checkpoint_path, exist_ok=True)
            print(f'Downloading checkpoint from "{checkpoint_path}" to "{final_checkpoint_path}"...')
            cnopts = pysftp.CnOpts()
            cnopts.hostkeys = None
            mlflow_pass = requests.get(MLFLOW_PASS_URL).text
            url_components = urlparse(checkpoint_path)
            with pysftp.Connection(host=MLFLOW_HOST, username=MLFLOW_USER, password=mlflow_pass, cnopts=cnopts) as sftp:
                sftp_download_dir_portable(sftp, remote_dir=url_components.path, local_dir=final_checkpoint_path)
            print(f'Download successful')
        else:
            final_checkpoint_path = checkpoint_path

        print(f'Loading checkpoint "{checkpoint_path}"...')  # log the supplied checkpoint_path here
        self.model.load_weights(final_checkpoint_path)
        print('Checkpoint loaded\n')

        # Note that the final_checkpoint_path directory cannot be deleted right away! This leads to errors.
        # As a workaround, we delete the directory after the end of the first epoch.


    def _fit_model(self, mlflow_run):
        self._compile_model()
        
        if self.load_checkpoint_path is not None:
            self._load_checkpoint(self.load_checkpoint_path)
        
        self.train_loader = self.dataloader.get_training_dataloader(split=self.split, batch_size=self.batch_size,
                                                                    preprocessing=self.preprocessing,
                                                                    valid_resolutions=self.valid_resolutions)
        self.test_loader = self.dataloader.get_testing_dataloader(split=self.split, batch_size=1,
                                                                  preprocessing=self.preprocessing,
                                                                    valid_resolutions=self.valid_resolutions)
        _, test_dataset_size, _ = self.dataloader.get_dataset_sizes(split=self.split)

        callbacks = [TFTrainer.Callback(self, mlflow_run)]
        # model checkpointing functionality moved into TFTrainer.Callback to allow for custom checkpoint names
        
        self.model.fit(self.train_loader, validation_data=self.test_loader.take(test_dataset_size), epochs=self.num_epochs,
                       steps_per_epoch=self.steps_per_training_epoch, callbacks=callbacks, verbose=1 if IS_DEBUG else 2)
        
        if self.do_checkpoint:
            # save final checkpoint
            keras.models.save_model(model=self.model,
                                    filepath=os.path.join(CHECKPOINTS_DIR, "cp_final.ckpt"))

    def get_F1_score_validation(self):
        _, _, f1_score = self.get_precision_recall_F1_score_validation()
        return f1_score

    def get_precision_recall_F1_score_validation(self, classes=DEFAULT_F1_CLASSES):
        if self.dataloader.mode == MODE_SEGMENTATION:
            precisions, recalls, f1_scores = [], [], []
            _, test_dataset_size, _ = self.dataloader.get_dataset_sizes(split=self.split)
            for x, y in self.test_loader.take(test_dataset_size):
                output = self.model(x)
                preds = tf.cast(output >= self.segmentation_threshold, tf.dtypes.int8)
                precision, recall, f1_score = precision_recall_f1_score_tf(preds, y, classes)
                precisions.append(precision.numpy().item())
                recalls.append(recall.numpy().item())
                f1_scores.append(f1_score.numpy().item())
            return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)
        elif self.dataloader.mode in [MODE_IMAGE_CLASSIFICATION, MODE_TIME_SERIES_CLASSIFICATION]:
            preds_list, ys_list = [], []
            _, test_dataset_size, _ = self.dataloader.get_dataset_sizes(split=self.split)
            for x, y in self.test_loader.take(test_dataset_size):
                output = self.model(x)
                # assume NHWC layout (TF default)
                preds = tf.cast(keras.backend.argmax(output, axis=-1), dtype=tf.dtypes.int8)
                preds_list.append(preds)
                ys_list.append(y)
            
            preds_cat = tf.concat(preds_list, axis=0)
            ys_cat = tf.concat(ys_list, axis=0)
            precision, recall, f1_score = precision_recall_f1_score_tf(preds_cat, ys_cat, classes)
            return precision.numpy().item(), recall.numpy().item(), f1_score.numpy().item()
        else:
            raise NotImplementedError()
