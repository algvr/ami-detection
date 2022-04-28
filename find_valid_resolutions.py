import json
import os
import tensorflow as tf
import tqdm

from models import *


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    model_class = ResNet50TF

    # we only use one channel here for speed reasons
    model = model_class(input_shape=(224, 224, 1))  # use 224x224 (works) just for initialization
    
    width_start = 16
    width_end = 3000
    width_step = 16
    
    height_start = 16
    height_end = 3000
    height_step = 16

    range_width = range(width_start, width_end + 1, width_step)
    range_height = range(height_start, height_end + 1, height_step)

    width_search_space_size = len(range_width)
    height_search_space_size = len(range_height)
    
    model_name = getattr(model, 'name', type(model).__name__)
    save_path = f'valid_resolutions_{model_name}.json'

    valid_resolutions = []

    iteration_idx = 0

    def save_progress(current_width, current_height, iteration_idx):
        with open(save_path, 'w') as f:
            info_obj = {'model': model_name,
                        'width_start': width_start, 'width_end': width_end, 'width_step': width_step, 
                        'height_start': height_start, 'height_end': height_end, 'height_step': height_step,
                        'covered_width': current_width, 'covered_height': current_height,
                        'iteration_idx': iteration_idx, 'valid_resolutions': valid_resolutions}
            f.write(json.dumps(info_obj))

    def load_progress():
        global iteration_idx, width_start, width_end, width_step, height_start, height_end, height_step,\
               range_width, range_height, valid_resolutions

        if not os.path.exists(save_path):
            return
        
        with open(save_path, 'r') as f:
            saved_config = json.loads(f.read())
        
        iteration_idx = saved_config['iteration_idx']
        print(f'\nResuming from iteration {iteration_idx}')

        if width_start != saved_config['width_start']:
            print(f"WARNING: width_start set to {width_start}, but saved config has {saved_config['width_start']}; "
                  f"using saved value")
            width_start = saved_config['width_start']
        
        if width_end != saved_config['width_end']:
            print(f"WARNING: width_end set to {width_end}, but saved config has {saved_config['width_end']}; "
                  f"using saved value")
            width_end = saved_config['width_end']
        
        if width_step != saved_config['width_step']:
            print(f"WARNING: width_step set to {width_step}, but saved config has {saved_config['width_step']}; "
                  f"using saved value")
            width_step = saved_config['width_step']
        
        if height_start != saved_config['height_start']:
            print(f"WARNING: height_start set to {height_start}, but saved config has {saved_config['height_start']}; "
                  f"using saved value")
            height_start = saved_config['height_start']
        
        if height_end != saved_config['height_end']:
            print(f"WARNING: height_end set to {height_end}, but saved config has {saved_config['height_end']}; "
                  f"using saved value")
            height_end = saved_config['height_end']
        
        if height_step != saved_config['height_step']:
            print(f"WARNING: height_step set to {height_step}, but saved config has {saved_config['height_step']}; "
                  f"using saved value")
            height_step = saved_config['height_step']
    
        covered_width = saved_config['covered_width']
        covered_height = saved_config['covered_height']

        valid_resolutions = saved_config['valid_resolutions']
        
        range_width = range(covered_width + width_step, width_end + 1, width_step)
        # this height range is valid only for the current width, and will be reset before the next width is processed
        range_height = range(covered_height + height_step, height_end + 1, height_step)

    
    load_progress()

    print('')
    with tqdm.tqdm(initial=iteration_idx, total=width_search_space_size * height_search_space_size) as progress_bar:
        for width in range_width:
            for height in range_height:
                x = tf.zeros((1, width, height, 1)) 
                try:
                    try:
                        del model
                    except:
                        pass
                    model = model_class(input_shape=(height, width, 1))
                    output = model(x)
                    valid_resolutions.append((width, height))
                except Exception as e:
                    pass
                if iteration_idx % 200 == 0:
                    save_progress(width, height, iteration_idx)
                progress_bar.update(1)
                iteration_idx += 1
            # reset height range
            range_height = range(height_start, height_end + 1, height_step)
        print('')
        print('Finished')
        save_progress(width, height, iteration_idx)
