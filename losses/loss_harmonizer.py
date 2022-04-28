# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

import keras
import keras.backend
import numpy as np
import tensorflow as tf
import torch


DEFAULT_TORCH_DIM_LAYOUT = 'NCHW'
DEFAULT_TF_DIM_LAYOUT = 'NHWC'


def collapse_channel_dim_torch(tensor, take_argmax, dim_layout=DEFAULT_TORCH_DIM_LAYOUT):
    shape = tensor.shape
    if len(shape) < 4:
        return tensor  # assume channel dim already collapsed
    channel_dim_idx = dim_layout.strip().upper().index('C')
    num_channels = shape[channel_dim_idx]
    if take_argmax and num_channels > 1:
        return torch.argmax(tensor, dim=channel_dim_idx).to(dtype=tensor.dtype)  # preserve original dtype
    else:
        # remove the channel dimension
        selector = [0 if dim_idx == channel_dim_idx else slice(0, shape[dim_idx]) for dim_idx in range(len(shape))]
        if take_argmax:
            return tensor[selector].round()
        else:
            return tensor[selector]


def expand_channel_dim_torch(tensor, channel_starts, dim_layout=DEFAULT_TORCH_DIM_LAYOUT):
    # channel_starts simultaneously determines the number of channels to add, and the starting thresholds of these
    # channels
    # * for a two-channel output, use channel_starts=(0.0, segmentation_threshold), e.g. (0.0, 0.5)
    # * for a rounded single-channel output (with 0.0 and 1.0), use channel_starts=(segmentation_threshold,)
    #   (a comma is needed for Python to interpret the channel_starts argument as a tuple)
    # * for a non-rounded single-channel output (equivalent to torch.unsqueeze with dim equal to the channel dimension),
    #   set channel_starts=None

    channel_dim_idx = dim_layout.strip().upper().index('C')
    # if channel_starts is None, prevent rounding
    if len(tensor.shape) == len(dim_layout) and channel_starts is not None:
        tensor = collapse_channel_dim_torch(tensor, take_argmax=True)

    if channel_starts is None:
        if len(tensor.shape) < len(dim_layout):
            tensor = torch.unsqueeze(tensor, dim=channel_dim_idx)
        return tensor

    tensor = torch.unsqueeze(tensor, dim=channel_dim_idx)
    repeat_selector = [len(channel_starts) if dim_idx == channel_dim_idx else 1 for dim_idx in range(len(tensor.shape))]
    tensor = tensor.repeat(*repeat_selector)
    for channel_idx in range(len(channel_starts)):
        selector = [channel_idx if dim_idx == channel_dim_idx else slice(0, tensor.shape[dim_idx])
                    for dim_idx in range(len(tensor.shape))]
        target_channel = tensor[selector]
        if channel_idx == len(channel_starts) - 1:
            mask = channel_starts[channel_idx] <= target_channel
        else:
            mask = torch.logical_and(channel_starts[channel_idx] <= target_channel,
                                     target_channel < channel_starts[channel_idx + 1])
        tensor[selector] = mask.to(dtype=tensor.dtype)
    return tensor


def collapse_channel_dim_tf(tensor, take_argmax, dim_layout=DEFAULT_TF_DIM_LAYOUT):
    shape = tensor.shape
    if len(shape) < 4:
        return tensor  # assume channel dim already collapsed
    channel_dim_idx = dim_layout.strip().upper().index('C')
    num_channels = shape[channel_dim_idx]
    if take_argmax and num_channels > 1:
        return tf.cast(keras.backend.argmax(tensor, axis=channel_dim_idx), dtype=tensor.dtype)  # preserve original dtype
    else:
        # remove the channel dimension
        selector = [0 if dim_idx == channel_dim_idx else slice(0, shape[dim_idx]) for dim_idx in range(len(shape))]
        if take_argmax:
            return tf.math.round(tensor[selector])
        else:
            return tensor[selector]


def expand_channel_dim_tf(tensor, channel_starts, dim_layout=DEFAULT_TF_DIM_LAYOUT):
    # the behavior of this function is identical to that of expand_channel_dim_torch; see there for a detailed
    # explanation of how to use this function

    channel_dim_idx = dim_layout.strip().upper().index('C')
    # if channel_starts is None, prevent rounding
    if len(tensor.shape) == len(dim_layout) and channel_starts is not None:
        tensor = collapse_channel_dim_tf(tensor, take_argmax=True)

    if channel_starts is None:
        if len(tensor.shape) < len(dim_layout):
            tensor = tf.expand_dims(tensor, axis=channel_dim_idx)
        return tensor

    tensor = tf.repeat(tf.expand_dims(tensor, axis=channel_dim_idx), repeats=len(channel_starts), axis=channel_dim_idx)

    channel_tensors = []
    for channel_idx in range(len(channel_starts)):
        # the selector here differs from the one in the torch-based function, as we need to preserve all dimensions
        # see comment about EagerTensor below for an explanation why
        selector = [slice(channel_idx, channel_idx + 1) if dim_idx == channel_dim_idx
                    else slice(0, tensor.shape[dim_idx])
                    for dim_idx in range(len(tensor.shape))]
        # here, we need to cast *target_channel*, as we may need to be able to compare target_channel a floating-point
        # value
        channel_start_tensor = tf.convert_to_tensor(channel_starts[channel_idx])
        target_channel = tf.cast(tensor[selector], dtype=channel_start_tensor.dtype)
        mask_geq = tf.math.greater_equal(target_channel, channel_start_tensor)
        if channel_idx == len(channel_starts) - 1:
            mask = mask_geq
        else:
            next_channel_start_tensor = tf.convert_to_tensor(channel_starts[channel_idx + 1])
            mask_l = tf.math.less(target_channel, next_channel_start_tensor)
            mask = tf.math.logical_and(mask_geq, mask_l)

        # TF's EagerTensor does not support item assignment, so we cannot do an in-place update
        # fix: create a separate Tensor for each channel, then stack tensors along channel dimension
        channel_tensors.append(tf.cast(mask, dtype=tensor.dtype))
    tensor = tf.concat(channel_tensors, axis=channel_dim_idx)
    return tensor
