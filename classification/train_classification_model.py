from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.data.all import *
from fastai.vision.all import *

from fastai.callback.core import *

from fastai.vision.widgets import *

import os
import pandas as pd
from time import time


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


path = os.path.join('datasets', 'ptb_v_classification_merged_aug', 'training')
datablock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock(vocab=['0', '1', '2'])),
                      get_items=get_image_files,
                      get_y=label_func,
                      splitter=RandomSplitter(valid_pct=0.1, seed=42))

batch_size = 12

dataloaders = datablock.dataloaders(path, bs=batch_size)

base_models = {'resnet152': models.resnet152}
base_model = 'resnet152'

learn = vision_learner(dataloaders, base_models[base_model], metrics=[recall_0, precision_0, f1_score_0,
                                                               recall_1, precision_1, f1_score_1,
                                                               recall_2, precision_2,
                                                               f1_score_2],
                                                               cbs=[CSVLogger(fname=f'results_{session_id}.csv'), CustomCallback()])


lr = 0.0003  # based on learn.lr_find()
num_epochs = 15
info_str = '\n'.join([f'session: {session_id}', f'base_model: {base_model}', f'lr: {lr}', f'batch_size: {batch_size}',
                      f'num_epochs: {num_epochs}'])
print(info_str)

with open(f'session_{session_id}_details.txt', 'w') as f:
    f.write(info_str)

learn.fine_tune(num_epochs, base_lr=lr)
learn.export('model_152_final.pkl')
