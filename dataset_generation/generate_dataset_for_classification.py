import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'segmentation'))


import numpy as np
from tqdm import tqdm

from segmentation.models import *
from request_processing.hough_processing import *
from request_processing import *



def process_generated(model, generate_time_series=False, classification_model_name=None, learner=None, json_path=None):
    objects, image_str = get_request_params_from_json(json_path)
    return process_request(segmentation_model=model,
                           objects=objects,
                           image_str=image_str,
                           token='placeholder',
                           original_sample_json_path=json_path,
                           generate_time_series=generate_time_series,
                           learner=learner,
                           classification_model_name=classification_model_name)


def process_generated_multi(model, generate_time_series=True, classification_model_name=None, path=None):
    if path is None:
        path = os.path.join('datasets', 'ptb_v', 'training')
    for folder_name, _, file_names in os.walk(path):
        for file_idx, file_name in tqdm(enumerate(file_names)):
            if file_name.lower().endswith('.json'):
                #try:
                    process_generated(model, generate_time_series, classification_model_name, None, os.path.join(path, file_name))
                #except Exception as e:
                #    print(f'Failed processing "{file_name}: {str(e)}"')


if __name__ == '__main__':
    # set segmentation model here
    segmentation_model = UNetTF(input_shape=(None, None, 3))


    # set name of classification model here (valid resolutions must be present in valid_res_lists)

    classification_model_name = "ResNet152TF"

    # !!!!! set checkpoint of segmentation model to load here !!!!!
    load_segmentation_model_checkpoint_path = os.path.join('downloaded_checkpoints', 'cp_ep-00002_it-05760_step-48000.ckpt')

    # change depending on whether this is a TF model or a Torch model
    segmentation_model.load_weights(load_segmentation_model_checkpoint_path)
    segmentation_model._checkpoint_path = load_segmentation_model_checkpoint_path
    
    process_generated_multi(segmentation_model, generate_time_series=False, classification_model_name=classification_model_name)
