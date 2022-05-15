# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

import keras.backend as K
import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf
import torch
from typing import Iterable

from .loss_harmonizer import *


DEFAULT_F1_CLASSES = (1, 2, 3)


def prediction_stats_torch(thresholded_prediction, targets, classes, dtype=torch.float32):
    targets = collapse_channel_dim_torch(targets, take_argmax=True).long()
    thresholded_prediction = collapse_channel_dim_torch(thresholded_prediction, take_argmax=True).long()

    tp, fp, tn, fn = [torch.zeros(1, dtype=dtype, device=targets.device) for _ in range(4)]

    if not isinstance(classes, Iterable):
        classes = [classes]

    for class_idx in classes:
        thr_pos = thresholded_prediction == class_idx
        thr_neg = thresholded_prediction != class_idx
        trg_pos = targets == class_idx
        trg_neg = targets != class_idx

        tp += torch.logical_and(thr_pos, trg_pos).sum()
        fp += torch.logical_and(thr_pos, trg_neg).sum()
        tn += torch.logical_and(thr_neg, trg_neg).sum()
        fn += torch.logical_and(thr_neg, trg_pos).sum()

    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def precision_torch(thresholded_prediction, targets, classes, pred_stats=None):
    if pred_stats is None:
        pred_stats = prediction_stats_torch(thresholded_prediction, targets, classes)
    if pred_stats['tp'] == torch.zeros_like(pred_stats['tp']):  # prevent NaNs
        return torch.zeros_like(pred_stats['tp'])
    return pred_stats['tp'] / (pred_stats['tp'] + pred_stats['fp'])


def recall_torch(thresholded_prediction, targets, classes, pred_stats=None):
    if pred_stats is None:
        pred_stats = prediction_stats_torch(thresholded_prediction, targets, classes)
    if pred_stats['tp'] == torch.zeros_like(pred_stats['tp']):  # prevent NaNs
        return torch.zeros_like(pred_stats['tp'])
    return pred_stats['tp'] / (pred_stats['tp'] + pred_stats['fn'])


def precision_recall_f1_score_torch(thresholded_prediction, targets, classes=DEFAULT_F1_CLASSES):
    # best value is at 1, worst at 0
    pred_stats = prediction_stats_torch(thresholded_prediction, targets, classes)
    precision = precision_torch(thresholded_prediction, targets, classes, pred_stats)
    recall = recall_torch(thresholded_prediction, targets, classes, pred_stats)
    if precision == torch.zeros_like(precision):  # prevent NaNs
        return precision, recall, torch.zeros_like(precision)
    f1_score = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1_score


def f1_score_torch(thresholded_prediction, targets, classes=DEFAULT_F1_CLASSES):
    _, _, f1_score = precision_recall_f1_score_torch(thresholded_prediction, targets, classes)
    return f1_score


def prediction_stats_tf(thresholded_prediction, targets, classes, dtype=tf.dtypes.float32):
    targets = collapse_channel_dim_tf(targets, take_argmax=True)
    thresholded_prediction = tf.cast(collapse_channel_dim_tf(thresholded_prediction, take_argmax=True),
                                     dtype=targets.dtype)

    tp, fp, tn, fn = [tf.zeros(1, dtype=dtype) for _ in range(4)]

    if not isinstance(classes, Iterable):
        classes = [classes]

    for class_idx in classes:
        class_idx_tensor = tf.cast(tf.convert_to_tensor(class_idx), dtype=targets.dtype)
        thr_pos = tf.math.equal(thresholded_prediction, class_idx_tensor)
        thr_neg = tf.math.not_equal(thresholded_prediction, class_idx_tensor)
        trg_pos = tf.math.equal(targets, class_idx_tensor)
        trg_neg = tf.math.not_equal(targets, class_idx_tensor)

        tp += tf.math.count_nonzero(tf.math.logical_and(thr_pos, trg_pos), dtype=tp.dtype)
        fp += tf.math.count_nonzero(tf.math.logical_and(thr_pos, trg_neg), dtype=fp.dtype)
        tn += tf.math.count_nonzero(tf.math.logical_and(thr_neg, trg_neg), dtype=tn.dtype)
        fn += tf.math.count_nonzero(tf.math.logical_and(thr_neg, trg_pos), dtype=fn.dtype)

    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def precision_tf(thresholded_prediction, targets, classes, pred_stats=None):
    if pred_stats is None:
        pred_stats = prediction_stats_tf(thresholded_prediction, targets, classes)
    if pred_stats['tp'] == tf.zeros_like(pred_stats['tp']):  # prevent NaNs
        return tf.zeros_like(pred_stats['tp'])
    return pred_stats['tp'] / (pred_stats['tp'] + pred_stats['fp'])


def recall_tf(thresholded_prediction, targets, classes, pred_stats=None):
    if pred_stats is None:
        pred_stats = prediction_stats_tf(thresholded_prediction, targets, classes)
    if pred_stats['tp'] == tf.zeros_like(pred_stats['tp']):  # prevent NaNs
        return tf.zeros_like(pred_stats['tp'])
    return pred_stats['tp'] / (pred_stats['tp'] + pred_stats['fn'])


def precision_recall_f1_score_tf(thresholded_prediction, targets, classes=DEFAULT_F1_CLASSES):
    # best value is at 1, worst at 0
    pred_stats = prediction_stats_tf(thresholded_prediction, targets, classes)
    precision = precision_tf(thresholded_prediction, targets, classes, pred_stats)
    recall = recall_tf(thresholded_prediction, targets, classes, pred_stats)
    if precision == tf.zeros_like(precision):  # prevent NaNs
        return precision, recall, tf.zeros_like(precision)
    f1_score = (tf.convert_to_tensor(2.0, dtype=precision.dtype) * precision * recall) / (precision + recall)
    return precision, recall, f1_score


def f1_score_tf(thresholded_prediction, targets, classes=DEFAULT_F1_CLASSES):
    _, _, f1_score = precision_recall_f1_score_tf(thresholded_prediction, targets, classes)
    return f1_score
