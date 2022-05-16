from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.data.all import *
from fastai.vision.all import *
from fastai.callback.core import *

from functools import partial 
import os
import pandas as pd
from time import time
import torch


def label_func(fname):
    return '1' if '_non_mi_abnormality_' in str(fname) else '2' if '_mi_' in str(fname) else '0'


session_id = int(time())


class CustomCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_idx = 0

    def after_epoch(self):
        self.learn.save(f'model_session={session_id}_epoch={self.epoch_idx}')
        self.epoch_idx += 1


recall_0 = RecallMulti(labels=[0])
precision_0 = PrecisionMulti(labels=[0])
f1_score_0 = F1ScoreMulti(labels=[0])

recall_1 = RecallMulti(labels=[1])
precision_1 = PrecisionMulti(labels=[1])
f1_score_1 = F1ScoreMulti(labels=[1])

recall_2 = RecallMulti(labels=[2])
precision_2 = PrecisionMulti(labels=[2])
f1_score_2 = F1ScoreMulti(labels=[2])

recall_1_2 = RecallMulti(labels=[1, 2])
precision_1_2 = PrecisionMulti(labels=[1, 2])
f1_score_1_2 = F1ScoreMulti(labels=[1, 2])

recall_total = RecallMulti(labels=[0, 1, 2])
precision_total = PrecisionMulti(labels=[0, 1, 2])
f1_score_total = F1ScoreMulti(labels=[0, 1, 2])


path_train = 'datasets/ptb_v_classification_merged_aug/training'

# removed the non-MI abnormality!
datablock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock(vocab=['0', '1', '2'])),
                      get_items=get_image_files,
                      get_y=label_func,
                      splitter=FuncSplitter(lambda path: 'validation' in str(path)))

batch_size = 12

dataloaders = datablock.dataloaders(path_train, bs=batch_size, verbose=True)

base_models = {'resnet152': models.resnet152}
base_model = 'resnet152'

loss_pre_eval = 'FocalLoss(weight=torch.tensor([1.0, 1.0, 1.5])' + ('.cuda()' if torch.cuda.is_available() else '') + ')'

learn = vision_learner(dataloaders, base_models[base_model], metrics=[recall_0, precision_0, f1_score_0,
                                                               recall_1, precision_1, f1_score_1,
                                                               recall_2, precision_2,
                                                               f1_score_2
                                                               ],
                                                               cbs=[CSVLogger(fname=f'results_{session_id}.csv'), CustomCallback()],
                                                               loss_func=eval(loss_pre_eval))


lr = 0.0003  # based on learn.lr_find()
num_epochs = 15
info_str = '\n'.join([f'session: {session_id}', f'base_model: {base_model}', f'lr: {lr}', f'batch_size: {batch_size}',
                      f'num_epochs: {num_epochs}', f'loss_fn: {loss_pre_eval}'])
print(info_str)

with open(f'session_{session_id}_details.txt', 'w') as f:
    f.write(info_str)

learn.fine_tune(num_epochs, base_lr=lr)
learn.export('model_152_final.pkl')
