import numpy as np
from tqdm import tqdm

from models import *
import os
from request_processing.hough_processing import *
from request_processing import *



def process_manually_segmented(model):
    with open('image_str.txt', 'r') as f:
        image_str = f.read()
    
    with open('image_objects.json', 'r') as f:
        objects = json.loads(f.read())

    return process_request(model, objects, image_str, 'placeholder')


def process_generated(model, generate_time_series=False, classification_model_name=None, learner=None, json_path=None):
    if json_path is None:
        json_path = 'datasets/ptb_v/training/01_17014_469374297_NORM=100.0_SR=0.0_data.json'
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
                try:
                    process_generated(model, generate_time_series, classification_model_name, os.path.join(path, file_name))
                except Exception as e:
                    print(f'Failed processing "{file_name}: {str(e)}"')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.cuda.is_available = lambda: False

    torch.set_default_tensor_type(torch.FloatTensor)

    # set segmentation model here
    segmentation_model = UNetTF(input_shape=(None, None, 3))
    # set classification model and its name here
    classification_model_name = 'ResNet152TF'


    datablock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock(vocab=['0', '1', '2'])),
                    get_items=get_image_files,
                    get_y=lambda *args, **kwargs: '0')

    dataloaders = datablock.dataloaders('datasets/', bs=2, device='cpu')

    learn = vision_learner(dataloaders, models.resnet152)

    state_dict_container = torch.load('saved_models/resnet152_ecg.pth', map_location=torch.device('cpu'))
    learn.model.load_state_dict(state_dict_container['model'])
    learn.model = learn.model.cpu()
    learn.model.eval()

    # set checkpoint to load here
    load_segmentation_model_checkpoint_path = os.path.join('downloaded_cps', 'cp_ep-00002_it-05760_step-48000.ckpt')
    # change depending on whether this is a TF model or a Torch model
    segmentation_model.load_weights(load_segmentation_model_checkpoint_path)
    segmentation_model._checkpoint_path = load_segmentation_model_checkpoint_path
    
    class_tp = [0, 0, 0]
    class_fp = [0, 0, 0]
    class_tn = [0, 0, 0]
    class_fn = [0, 0, 0]
    class_total = [0, 0, 0]
    total_preds = 0

    json_paths = []
    path = os.path.join('datasets', 'ptb_v', 'training')
    for folder_name, _, file_names in os.walk(path):
        for file_idx, file_name in tqdm(enumerate(file_names)):
            if file_name.lower().endswith('.json'):
                json_paths.append(os.path.join(path, file_name))

    for json_path in json_paths:
        with open(json_path, 'r') as json_file:
            json_config = json.loads(json_file.read())
            preds = process_generated(model=segmentation_model,
                                      json_path=json_path,
                                      generate_time_series=False,
                                      classification_model_name=classification_model_name,
                                      learner=learn)
            pred_class = np.argmax([preds['normal'], preds['non-mi-related-abnormalities'], preds['mi']])
            true_class = 1 if json_config['_is_non_mi_abnormality'] else 2 if json_config['_is_mi'] else 0
            print(f'Prediction: {pred_class}; ground-truth: {true_class}')
            class_total[true_class] += 1
            total_preds += 1
            if pred_class == true_class:
                class_tp[pred_class] += 1
                for _class in [0, 1, 2]:
                    if _class != pred_class:
                        class_tn[_class] += 1
            else:
                class_fp[pred_class] += 1
                class_fn[true_class] += 1
                for _class in [0, 1, 2]:
                    if _class not in [pred_class, true_class]:
                        class_tn[_class] += 1

    precisions = [class_tp[_class] / (class_tp[_class] + class_fp[_class]) if class_tp[_class] + class_fp[_class] > 0 else 0 for _class in [0, 1, 2]]
    recalls = [class_tp[_class] / (class_tp[_class] + class_fn[_class]) if class_tp[_class] + class_fn[_class] > 0 else 0 for _class in [0, 1, 2]]

    print(f'*** Stats: ***')
    for _class in [0, 1, 2]:
        print(f'Class "{["normal", "non_mi_abnormality", "mi"][_class]}": precision {"%.4f" % precisions[_class]}; recall {"%.4f" % recalls[_class]}')
        

    #while True:
    #    process_generated_multi(segmentation_model, generate_time_series=False,
    #                            classification_model_name=classification_model_name)
