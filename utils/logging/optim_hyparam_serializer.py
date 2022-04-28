import keras.optimizers
import tensorflow as tf
import torch.optim as T
import tensorflow.keras.optimizers as K


# there could be multiple names describing the same param, but we want the names to be consistent so we can
# compare the parameters in MLflow easily
# make sure to keep the names consistent!


def _get_consistent_hyparam_name(name):
    # ensure consistency of all hyparam names with Torch naming scheme
    # keys: target names; values: names to rename
    rename_dir = {
        'lr': ['learning_rate'],
        'eps': ['epsilon']
    }
    for key in rename_dir.keys():
        if name in rename_dir[key]:
            return 'opt_' + key

    return 'opt_' + name


def serialize_optimizer_hyperparams(optimizer_object):
    """
    Args:
        optimizer_object: optimizer to serialize
    Returns:
        dict containing hyperparameters describing the given optimizer_object;
        note that the optimizer may contain actual parameters (e.g. m and v in Adam) that will not be part of the result
    """

    # Torch optimizers

    _ser_T = _serialize_generic_torch_hyparams
    _ser_T_adam = _serialize_torch_adam_hyparams

    serialization_functions_for_types_T = {
        T.Adadelta:     lambda opt: _ser_T(opt, 'Adadelta',
                                           ['lr', 'rho', 'eps', 'weight_decay']),
        T.Adagrad:      lambda opt: _ser_T(opt, 'Adagrad',
                                           ['lr', 'lr_decay', 'weight_decay', 'initial_accumulator_value', 'eps']),
        T.Adam:         _serialize_torch_adam_hyparams,
        T.AdamW:        lambda opt: _serialize_torch_adam_hyparams(opt, 'AdamW',
                                           ['lr', 'eps', 'weight_decay', 'adamw_amsgrad=amsgrad']),
        T.SparseAdam:   lambda opt: _ser_T_adam(opt, 'SparseAdam',
                                           ['lr', 'eps']),
        T.Adamax:       lambda opt: _ser_T_adam(opt, 'Adamax',
                                           ['lr', 'eps', 'weight_decay']),
        T.ASGD:         lambda opt: _ser_T(opt, 'ASGD',
                                           ['lr', 'lambda=lambd', 'alpha', 't0', 'weight_decay']),
        T.LBFGS:        lambda opt: _ser_T(opt, 'LBFGS',
                                           ['lr', 'max_iter', 'max_eval', 'tolerance_grad', 'tolerance_change',
                                            'history_size']),
        T.NAdam:        lambda opt: _ser_T_adam(opt, 'NAdam',
                                           ['lr', 'eps', 'weight_decay', 'momentum_decay']),
        T.RAdam:        lambda opt: _ser_T_adam(opt, 'RAdam',
                                           ['lr', 'eps', 'weight_decay']),
        T.RMSprop:      lambda opt: _ser_T(opt, 'RMSprop',
                                           ['lr', 'alpha', 'eps', 'weight_decay', 'momentum',
                                            'rmsprop_centered=centered']),
        T.Rprop:        _serialize_torch_rprop_hyparams,
        # omitted hyparams for T.SGD: maximize
        T.SGD:          lambda opt: _ser_T(opt, 'SGD',
                                           ['lr', 'momentum', 'dampening', 'weight_decay', 'sgd_nesterov=nesterov'])
    }

    # Keras optimizers

    _ser_K = _serialize_generic_keras_hyparams

    serialization_functions_for_types_K = {
        K.SGD:          lambda opt: _ser_K(opt, 'SGD',
                                           ['learning_rate', 'momentum', 'nesterov']),
        K.RMSprop:      lambda opt: _ser_K(opt, 'RMSprop',
                                           ['learning_rate', 'rho', 'momentum', 'epsilon',
                                            'rmsprop_centered=centered']),
        K.Adam:         lambda opt: _ser_K(opt, 'Adam',
                                           ['learning_rate', 'beta_1', 'beta_2', 'epsilon', 'amsgrad']),
        K.Adadelta:     lambda opt: _ser_K(opt, 'Adadelta',
                                           ['learning_rate', 'rho', 'epsilon']),
        K.Adagrad:      lambda opt: _ser_K(opt, 'Adagrad',
                                           ['learning_rate', 'initial_accumulator_value', 'epsilon']),
        K.Adamax:       lambda opt: _ser_K(opt, 'Adamax',
                                           ['learning_rate', 'beta_1', 'beta_2', 'epsilon']),
        K.Nadam:        lambda opt: _ser_K(opt, 'Nadam',
                                           ['learning_rate', 'beta_1', 'beta_2', 'epsilon']),
        K.Ftrl:         lambda opt: _ser_K(opt, 'Ftrl',
                                           ['learning_rate', 'learning_rate_power', 'initial_accumulator_value',
                                            'l1_regularization_strength', 'l2_regularization_strength',
                                            'l2_shrinkage_regularization_strength', 'beta'])
    }

    serialization_functions_for_types = {**serialization_functions_for_types_T, **serialization_functions_for_types_K}

    # if type(optimizer_object) not in serialization_functions_for_types, empty dict will be returned
    if type(optimizer_object) in serialization_functions_for_types:
        return serialization_functions_for_types[type(optimizer_object)](optimizer_object)

    return {'optimizer_str': str(optimizer_object)}


def _get_torch_hyparam_name(name, opt, param_group_idx):
    suffix = '' if len(opt.param_groups) <= 1 else f'__{param_group_idx}'
    return _get_consistent_hyparam_name(name) + suffix


def _get_torch_hyparam(name, opt, param_group_idx):
    return opt.param_groups[param_group_idx][name]


def _serialize_generic_torch_hyparams(opt, opt_name, hyparams):
    # can be used whenever an optimizer has no hyperparameters requiring "special treatment"

    output = {'optimizer': opt_name}
    for param_group_idx, param_group in enumerate(opt.param_groups):
        # dict to reduce code duplication
        p = {'opt': opt,
             'param_group_idx': param_group_idx}

        # simply copy to output
        for hyparam in hyparams:
            eq_idx = hyparam.find('=')
            if eq_idx > -1:
                input_name = hyparam[eq_idx+1:].strip()
                output_name = hyparam[:eq_idx].strip()
            else:
                input_name, output_name = hyparam, hyparam
            output[_get_torch_hyparam_name(output_name, **p)] = _get_torch_hyparam(input_name, **p)
    return output


def _serialize_torch_adam_hyparams(opt, opt_name='Adam', generic_params=['lr', 'eps', 'weight_decay', 'amsgrad']):
    # we need this function to break apart the "beta" hyparam of Adam-based optimizers into two separate numbers
    # omitted hyparams: maximize

    output = _serialize_generic_torch_hyparams(opt, opt_name, generic_params)

    # special treatment
    for param_group_idx, param_group in enumerate(opt.param_groups):
        # dict to reduce code duplication
        p = {'opt': opt,
             'param_group_idx': param_group_idx}
        output[_get_torch_hyparam_name('beta_1', **p)] = _get_torch_hyparam('betas', **p)[0]
        output[_get_torch_hyparam_name('beta_2', **p)] = _get_torch_hyparam('betas', **p)[1]

    return output


def _serialize_torch_rprop_hyparams(opt, opt_name='RProp', generic_params=['lr']):
    # we need this function to break apart the "eta" and "step_sizes" hyparams of RProp-based optimizers into
    # two separate numbers each

    output = _serialize_generic_torch_hyparams(opt, opt_name, generic_params)

    # special treatment
    for param_group_idx, param_group in enumerate(opt.param_groups):
        # dict to reduce code duplication
        p = {'opt': opt,
             'param_group_idx': param_group_idx}
        output[_get_torch_hyparam_name('eta_1', **p)] = _get_torch_hyparam('etas', **p)[0]
        output[_get_torch_hyparam_name('eta_2', **p)] = _get_torch_hyparam('etas', **p)[1]
        output[_get_torch_hyparam_name('step_min', **p)] = _get_torch_hyparam('step_sizes', **p)[0]
        output[_get_torch_hyparam_name('step_max', **p)] = _get_torch_hyparam('step_sizes', **p)[1]

    return output


def _get_keras_hyparam_name(name, opt):
    return _get_consistent_hyparam_name(name)


def _get_keras_hyparam(name, opt):
    value = getattr(opt, name)
    if isinstance(value, tf.Variable):
        return value.numpy()
    return value


def _serialize_generic_keras_hyparams(opt, opt_name, hyparams):
    output = {'optimizer': opt_name}
    # simply copy to output
    for hyparam in hyparams:
        eq_idx = hyparam.find('=')
        if eq_idx > -1:
            input_name = hyparam[eq_idx+1:].strip()
            output_name = hyparam[:eq_idx].strip()
        else:
            input_name, output_name = hyparam, hyparam
        output[_get_keras_hyparam_name(output_name, opt)] = _get_keras_hyparam(input_name, opt)
    return output
