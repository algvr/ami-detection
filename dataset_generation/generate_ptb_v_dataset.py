import datetime
from data_augmentation import *
from IPython.core.display_functions import display
import json
import math
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


# create dataset directories, set save_dir

root_dir = os.path.join('datasets', 'ptb_v')
config_dir = root_dir
save_dir = os.path.join(root_dir, 'training')
for dir_path in [root_dir, config_dir, save_dir]:
    os.makedirs(dir_path, exist_ok=True)


sample_idx_start = 0
num_samples_to_generate = 2000

for sample_idx in tqdm(range(sample_idx_start, sample_idx_start + num_samples_to_generate)):
    photo, mask_imgs, data, lead_pos, angle = get_random_ecg_photo()
    scp_codes = data['scp_codes']
    scp_code_str = '_'.join([f'{k}={v}' for k, v in scp_codes.items()])
    rec_date = datetime.datetime.strptime(data['recording_date'], '%Y-%m-%d %H:%M:%S')
    rec_timestamp = (rec_date - datetime.datetime(1970, 1, 1)).total_seconds()
    id_str = str(int(data['patient_id'])) + '_' + str(int(rec_timestamp))
    if len(scp_code_str) > 0:
        scp_code_str = '_' + scp_code_str
    path_no_ext = (('%0' + str(math.ceil(math.log10(num_samples_to_generate))) + 'd_' + id_str + scp_code_str)
                    % sample_idx).replace('/', '-').replace('\\', '-')
    path_photo = os.path.join(save_dir, path_no_ext + '.jpg')
    photo.convert('RGB').save(path_photo)
    
    # now, we can merge the individual masks
    # use channel 3 (alpha channel) to detect presence of pixels

    curve_mask_arr = (np.array(mask_imgs[0])[:, :, 3] > 0).astype(np.uint8)
    thick_hor_lines_mask_arr = (np.array(mask_imgs[1])[:, :, 3] > 0).astype(np.uint8)
    thick_vert_lines_mask_arr = (np.array(mask_imgs[2])[:, :, 3] > 0).astype(np.uint8)

    mask_arr = ((thick_hor_lines_mask_arr * 85 * (1 - thick_vert_lines_mask_arr) + thick_vert_lines_mask_arr * 170) * 
                (1 - curve_mask_arr)) + curve_mask_arr * 255

    path_mask_img = os.path.join(save_dir, path_no_ext + '_mask.png')
    mask_img = Image.fromarray(mask_arr)
    mask_img.save(path_mask_img)

    img_width, img_height = photo.size
    with open(os.path.join(save_dir, path_no_ext + '_data.json'), 'w') as lead_pos_file:
        lead_pos['_img_path'] = path_photo
        lead_pos['_mask_img_path'] = path_mask_img
        lead_pos['_angle'] = angle
        lead_pos['_img_width'] = img_width
        lead_pos['_img_height'] = img_height
        lead_pos['_scp_codes'] = data['scp_codes']

        # set label
        
        mi_list = ['IMI', 'ASMI', 'ILMI', 'AMI', 'LMI', 'IPLMI', 'IPMI', 'PMI']
        if len(set(data['scp_codes'].keys()) & set(mi_list)) > 0:
            is_normal, is_non_mi_abnormality, is_mi = [False, False, True]
        elif len(set(data['scp_codes'].keys()) & set(['NORM'])) == 1 and data['scp_codes']['NORM'] >= 80.0:
            is_normal, is_non_mi_abnormality, is_mi = [True, False, False]
        else:
            is_normal, is_non_mi_abnormality, is_mi = [False, True, False]
        
        lead_pos['_is_normal'] = is_normal
        lead_pos['_is_non_mi_abnormality'] = is_non_mi_abnormality
        lead_pos['_is_mi'] = is_mi

        lead_pos_file.write(json.dumps(lead_pos))

# write dataset parameters

dataset_params = get_default_ecg_param_dict()
dataset_params['num_samples'] = num_samples_to_generate
with open(os.path.join(config_dir, 'dataset_hyperparams.json'), 'w') as f:
    f.write(json.dumps(dataset_params))
