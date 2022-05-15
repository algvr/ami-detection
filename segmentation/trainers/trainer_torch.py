# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

import abc
import datetime
import functools
import math
import numpy as np
import pysftp
import requests
from sklearn.utils import shuffle
import torch
import torch.cuda
from torch.utils.data import DataLoader, Subset
from urllib.parse import urlparse

from losses.precision_recall_f1 import *
from utils.logging import mlflow_logger
from .trainer import Trainer
from utils import *


class TorchTrainer(Trainer, abc.ABC):
    def __init__(self, dataloader, model, preprocessing,
                 experiment_name=None, run_name=None, split=None, num_epochs=None, batch_size=None,
                 optimizer_or_lr=None, scheduler=None, loss_function=None, loss_function_hyperparams=None,
                 evaluation_interval=None, num_samples_to_visualize=None, checkpoint_interval=None,
                 load_checkpoint_path=None, segmentation_threshold=None):
        """
        Abstract class for Torch-based model trainers.
        Args:
            dataloader: the DataLoader to use when training the model
            model: the model to train
        """
        super().__init__(dataloader, model, experiment_name, run_name, split, num_epochs, batch_size, optimizer_or_lr,
                         loss_function, loss_function_hyperparams, evaluation_interval, num_samples_to_visualize,
                         checkpoint_interval, load_checkpoint_path, segmentation_threshold)
        # these attributes must also be set by each TFTrainer subclass upon initialization:
        self.preprocessing = preprocessing
        self.scheduler = scheduler

    # This class is a mimicry of TensorFlow's "Callback" class behavior, used not out of necessity (as we write the
    # training loop explicitly in Torch, instead of using the "all-inclusive" model.fit(...) in TF, and can thus just
    # build event handlers into the training loop), but to have a more consistent codebase across TorchTrainer and
    # TFTrainer.
    # See the comment next to the KC.Callback subclass in "trainer_tf.py" for more information.
    class Callback:
        def __init__(self, trainer, mlflow_run, model):
            super().__init__()
            self.model = model
            self.trainer = trainer
            self.mlflow_run = mlflow_run
            self.do_evaluate = self.trainer.evaluation_interval is not None and self.trainer.evaluation_interval > 0
            self.iteration_idx = 0
            self.epoch_idx = 0
            self.epoch_iteration_idx = 0
            self.do_visualize = self.trainer.num_samples_to_visualize is not None and \
                                self.trainer.num_samples_to_visualize > 0

        def on_train_batch_end(self):
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
                      % (self.iteration_idx, self.epoch_idx, self.epoch_iteration_idx, str(metrics)))
                if mlflow_logger.logging_to_mlflow_enabled():
                    mlflow_logger.log_metrics(metrics, aggregate_iteration_idx=self.iteration_idx)
                    if self.do_visualize and self.dataloader.mode == MODE_SEGMENTATION:
                        mlflow_logger.log_visualizations(self.trainer, self.iteration_idx)
                
                if self.trainer.do_checkpoint\
                   and self.iteration_idx % self.trainer.checkpoint_interval == 0\
                   and self.iteration_idx > 0:  # avoid creating checkpoints at iteration 0
                    self.trainer._save_checkpoint(self.trainer.model, self.epoch_idx, self.epoch_iteration_idx, self.iteration_idx)

            self.iteration_idx += 1
            self.epoch_iteration_idx += 1
    
        def on_epoch_end(self):
            self.epoch_idx += 1
            self.epoch_iteration_idx = 0

    # Visualizations are created using mlflow_logger's "log_visualizations" (containing ML framework-independent code),
    # and the "create_visualizations" functions of the Trainer subclasses (containing ML framework-specific code)
    # Specifically, the Trainer calls mlflow_logger's "log_visualizations" (e.g. in "on_train_batch_end" of the
    # tensorflow.keras.callbacks.Callback subclass), which in turn uses the Trainer's "create_visualizations".
    def create_visualizations(self, file_path):
        # sample image indices to visualize
        # fix half of the samples, randomize other half
        # the first, fixed half of samples serves for comparison purposes across models/runs
        # the second, randomized half allows us to spot weaknesses of the model we might miss when
        # always visualizing the same samples
        num_to_visualize = self.num_samples_to_visualize
        num_fixed_samples = num_to_visualize // 2
        num_random_samples = num_to_visualize - num_fixed_samples
        # start sampling random indices from "num_fixed_samples + 1" to avoid duplicates
        # convert to np.array to allow subsequent slicing
        if num_to_visualize >= len(self.test_loader):
            # just visualize the entire test set
            indices = np.array(list(range(len(self.test_loader))))
        else:
            indices = np.array(list(range(num_fixed_samples)) +\
                            random.sample(range(num_fixed_samples + 1, len(self.test_loader)), num_random_samples))
        images = []
        # never exceed the given training batch size, else we might face memory problems
        vis_batch_size = min(num_to_visualize, self.batch_size)
        subset_ds = Subset(self.test_loader.dataset, indices)
        subset_dl = DataLoader(subset_ds, batch_size=vis_batch_size, shuffle=False)
        
        for (batch_xs, batch_ys) in subset_dl:
            batch_xs, batch_ys = batch_xs.to(self.device), batch_ys.numpy()
            if len(batch_ys.shape) < 4:
                batch_ys = torch.unsqueeze(batch_ys, dim=-1)
            output = self.model(batch_xs)
            output_argmax = torch.argmax(output, dim=DEFAULT_TORCH_DIM_LAYOUT.find('C'), keepdim=True)
            preds = output_argmax.float().cpu().detach().numpy()
            # At this point we should have preds.shape = (batch_size, 1, H, W) and same for batch_ys
            self._fill_images_array(preds, batch_ys, images)

        self._save_image_array(images, file_path)

    def _save_checkpoint(self, model, epoch, epoch_iteration, total_iteration):
        if None not in [epoch, epoch_iteration, total_iteration]:
            checkpoint_path = f'{CHECKPOINTS_DIR}/cp_ep-{"%05i" % epoch}_epit-{"%05i" % epoch_iteration}' +\
                              f'_step-{total_iteration}.pt'
        else:
            checkpoint_path = f'{CHECKPOINTS_DIR}/cp_final.pt'
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': self.optimizer_or_lr.state_dict()
        }, checkpoint_path)
        
        # checkpoints should be logged to MLflow right after their creation, so that if training is
        # stopped/crashes *without* reaching the final "mlflow_logger.log_checkpoints()" call in trainer.py,
        # prior checkpoints have already been persisted
        mlflow_logger.log_checkpoints()

    def _load_checkpoint(self, checkpoint_path):
        # this function may only be called after self.device was initialized, and after self.model has been moved
        # to self.device

        print(f'\n*** WARNING: resuming training from checkpoint "{checkpoint_path}" ***\n')
        load_from_sftp = checkpoint_path.lower().startswith('sftp://')
        if load_from_sftp:
            final_checkpoint_path = f'original_checkpoint_{SESSION_ID}.pt'
            print(f'Downloading checkpoint from "{checkpoint_path}" to "{final_checkpoint_path}"...')
            cnopts = pysftp.CnOpts()
            cnopts.hostkeys = None
            mlflow_pass = requests.get(MLFLOW_PASS_URL).text
            url_components = urlparse(checkpoint_path)
            with pysftp.Connection(host=MLFLOW_HOST, username=MLFLOW_USER, password=mlflow_pass, cnopts=cnopts) as sftp:
                sftp.get(url_components.path, final_checkpoint_path)
            print(f'Download successful')
        else:
            final_checkpoint_path = checkpoint_path

        print(f'Loading checkpoint "{checkpoint_path}"...')  # log the supplied checkpoint_path here
        checkpoint = torch.load(final_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer_or_lr.load_state_dict(checkpoint['optimizer'])
        print('Checkpoint loaded\n')
        os.remove(final_checkpoint_path)

    def _fit_model(self, mlflow_run):
        print('\nTraining started at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        print(f'Session ID: {SESSION_ID}')
        print('Hyperparameters:')
        print(self._get_hyperparams())
        print('')

        self.train_loader = self.dataloader.get_training_dataloader(split=self.split, batch_size=self.batch_size,
                                                                    preprocessing=self.preprocessing,
                                                                    valid_resolutions=self.valid_resolutions)
        self.test_loader = self.dataloader.get_testing_dataloader(batch_size=1,
                                                                  preprocessing=self.preprocessing,
                                                                  valid_resolutions=self.valid_resolutions)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f'Using device: {self.device}\n')

        self.model = self.model.to(self.device)

        if self.load_checkpoint_path is not None:
            self._load_checkpoint(self.load_checkpoint_path)

        callback_handler = TorchTrainer.Callback(self, mlflow_run, self.model)
        for epoch in range(self.num_epochs):
            last_train_loss = self._train_step(self.model, self.device, self.train_loader,
                                               callback_handler=callback_handler)
            last_test_loss = self._eval_step(self.model, self.device, self.test_loader)
            metrics = {'train_loss': last_train_loss, 'test_loss': last_test_loss}

            print('\nEpoch %i finished at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) % epoch)
            print('Metrics: %s\n' % str(metrics))

            mlflow_logger.log_metrics(metrics, aggregate_iteration_idx=callback_handler.iteration_idx)
            mlflow_logger.log_logfiles()
        if self.do_checkpoint:
            # save final checkpoint
            self._save_checkpoint(self.model, None, None, None)
        
        print('\nTraining finished at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        mlflow_logger.log_logfiles()
        return last_test_loss

    def _train_step(self, model, device, train_loader, callback_handler):
        # WARNING: some models subclassing TorchTrainer overwrite this function, so make sure any changes here are
        # reflected appropriately in these models' files
        model.train()
        opt = self.optimizer_or_lr
        train_loss = 0
        for (x, y) in train_loader:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)
            y = torch.squeeze(y, dim=1)  # y must be of shape (batch_size, H, W) not (batch_size, 1, H, W)
            preds = model(x)
            loss = self.loss_function(preds, y)
            with torch.no_grad():
                train_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            callback_handler.on_train_batch_end()
            del x
            del y
        train_loss /= len(train_loader.dataset)
        callback_handler.on_epoch_end()
        self.scheduler.step()
        return train_loss

    def _eval_step(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for (x, y) in test_loader:
                x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)
                y = torch.squeeze(y, dim=1)
                preds = model(x)
                test_loss += self.loss_function(preds, y).item()
                del x
                del y
        test_loss /= len(test_loader.dataset)
        return test_loss

    def get_F1_score_validation(self):
        _, _, f1_score = self.get_precision_recall_F1_score_validation()
        return f1_score

    def get_precision_recall_F1_score_validation(self, classes=DEFAULT_F1_CLASSES):
        self.model.eval()
        preds_list, ys_list = [], []
        for (x, y) in self.test_loader:
            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.float32)
            output = self.model(x)
            # assume NCHW layout (Torch default)
            preds = torch.argmax(output, axis=1)
            preds_list.append(preds)
            ys_list.append(y)
            del x
            del y

        preds_cat = torch.cat(preds_list, dim=0)
        ys_cat = torch.cat(ys_list, dim=0)
        precision, recall, f1_score = precision_recall_f1_score_torch(preds_cat, ys_cat, classes)
        return precision.cpu().detach().numpy().item(),\
               recall.cpu().detach().numpy().item(),\
               f1_score.cpu().detach().numpy().item()
