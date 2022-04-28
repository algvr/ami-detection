# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

import math
import tensorflow as tf
import tensorflow.keras as K

from .trainer_tf import TFTrainer
from utils import *


class UNetTFTrainer(TFTrainer):
    """
    Trainer for the UnetTF model.
    """

    def __init__(self, dataloader, model, experiment_name=None, run_name=None, split=None, num_epochs=None,
                 batch_size=None, optimizer_or_lr=None, loss_function=None, loss_function_hyperparams=None,
                 evaluation_interval=None, num_samples_to_visualize=None, checkpoint_interval=None,
                 load_checkpoint_path=None, segmentation_threshold=None):
        # set omitted parameters to model-specific defaults, then call superclass __init__ function
        # warning: some arguments depend on others not being None, so respect this order!

        if split is None:
            split = DEFAULT_TRAIN_FRACTION

        # Large batch size used online: 32
        # Possible overkill
        # WARNING: cannot trivially use a batch size > 1, since in general, we have samples of different sizes!
        if batch_size is None:
            batch_size = 1

        train_set_size, test_set_size, unlabeled_test_set_size = dataloader.get_dataset_sizes(split=split)
        steps_per_training_epoch = train_set_size // batch_size

        if num_epochs is None:
            num_epochs = math.ceil(100000 / steps_per_training_epoch)

        if optimizer_or_lr is None:
            # CAREFUL! Smaller learning rate recommended in comparision to other models !!!
            # Even 1e-5 was recommended, but might take ages
            optimizer_or_lr = UNetTFTrainer.get_default_optimizer_with_lr(lr=1e-4)
        elif isinstance(optimizer_or_lr, int) or isinstance(optimizer_or_lr, float):
            optimizer_or_lr = UNetTFTrainer.get_default_optimizer_with_lr(lr=optimizer_or_lr)

        # According to the online github repo
        if loss_function is None:
            loss_function = K.losses.SparseCategoricalCrossentropy(from_logits=False,
                                                                   reduction=K.losses.Reduction.SUM_OVER_BATCH_SIZE)

        if evaluation_interval is None:
            evaluation_interval = dataloader.get_default_evaluation_interval(split, batch_size, num_epochs, num_samples_to_visualize)

        # convert model input to float32 \in [0, 1] & remove A channel;
        # convert ground truth to int \in [0, 3] & collapse channel dimension
        # classes: {0: 'bg', 1: 'thick_hor_lines', 2: 'thick_vert_lines', 3: 'ecg_curve'}
        preprocessing =\
            lambda x, is_gt: (tf.cast(x[:, :, :3], dtype=tf.float32) / 255.0) if not is_gt \
            else (x[:, :, 0] // 85)

        super().__init__(dataloader, model, preprocessing, steps_per_training_epoch, experiment_name, run_name, split,
                         num_epochs, batch_size, optimizer_or_lr, loss_function, loss_function_hyperparams,
                         evaluation_interval, num_samples_to_visualize, checkpoint_interval, load_checkpoint_path,
                         segmentation_threshold)

    def _get_hyperparams(self):
        return {**(super()._get_hyperparams()),
                **({param: getattr(self.model, param)
                   for param in ['dropout', 'kernel_init', 'normalize', 'kernel_regularizer', 'up_transpose']
                   if hasattr(self.model, param)})}
    
    @staticmethod
    def get_default_optimizer_with_lr(lr):
        # no mention on learning rate decay; can be reintroduced
        # lr_schedule = K.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_rate=0.1,
        #                                                      decay_steps=30000, staircase=True)
        return K.optimizers.Adam(learning_rate=lr)
