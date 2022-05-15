import numpy as np
import os
from PIL import Image, ImageDraw
from tqdm import tqdm


CLASSIFICATION_NETWORK_IMG_WIDTH = 640
CLASSIFICATION_NETWORK_LEAD_HEIGHT = 120  # height for a single lead


if __name__ == '__main__':
    ptb_v_classification_dir = os.path.join('datasets', 'ptb_v_classification', 'training')
    ptb_v_classification_aug_dir = os.path.join('datasets', 'ptb_v_classification_aug', 'training')

    num_augmented_files_per_sample = 3
    mix_leads = True

    os.makedirs(ptb_v_classification_aug_dir, exist_ok=True)

    # inspection revealed: most ECGs only present in dataset once (makes sense: < 20k recordings visualized)

    images_by_ecg = {}
    images_by_class_and_channel = {}

    num_distinct_imgs = 0

    if os.path.isdir(ptb_v_classification_dir):
        for folder_name, _, filenames in os.walk(ptb_v_classification_dir):
            # determine last index
            largest_idx = -1
            for filename in filter(lambda path: 'channel=0' in path.lower(), filenames):
                try:
                    sample_idx = int(filename.split('_')[0])

                    file_parts = filename.split('__')
                    pat_rec_id = '_'.join(file_parts[2].split('_')[:2])

                    sample_class = file_parts[1]

                    file_path = os.path.join(ptb_v_classification_dir, filename)
                    if pat_rec_id in images_by_ecg:
                        images_by_ecg[pat_rec_id].append(file_path)
                    else:
                        images_by_ecg[pat_rec_id] = [file_path]

                    for sample_class, channel in [(sample_class, '0'), (sample_class, '1'), (sample_class, '2')]:
                        key = f'{sample_class}_{channel}'
                        if not os.path.isfile(file_path.replace('channel=0', f'channel={channel}')):
                            continue
                        if key in images_by_class_and_channel:
                            images_by_class_and_channel[key].append(file_path)
                        else:
                            images_by_class_and_channel[key] = [file_path]
                    
                    if sample_idx > largest_idx:
                        largest_idx = sample_idx

                    num_distinct_imgs += 1
                except Exception as e:
                    continue
            
            next_aug_idx = largest_idx + 1


    for filename in tqdm(filter(lambda path: 'channel=0' in path.lower(), filenames)):
        fn_root, ext = os.path.splitext(filename)
        if ext.lower() != '.png':
            continue
        # determine channel files associated with this
        
        try:
            sample_idx = int(filename.split('_')[0])

            file_parts = filename.split('__')
            pat_rec_id = '_'.join(file_parts[2].split('_')[:2])

            sample_class = file_parts[1]

            file_path = os.path.join(ptb_v_classification_dir, filename)
            channel_paths = [os.path.join(ptb_v_classification_dir, '='.join(filename.split('=')[:-1])
                                                    + f'={channel_idx}{ext}') for channel_idx in range(3)]

            if any(filter(lambda path: not os.path.isfile(path), channel_paths)):
                continue

            # perform the lead swap before stretching the leads
            for aug_idx in range(num_augmented_files_per_sample):
                # swap two leads per channel
                for path in channel_paths:
                    path_fn_root, path_ext = os.path.splitext(os.path.basename(path))
                    with Image.open(path) as orig_img:
                        orig_draw = ImageDraw.Draw(orig_img)
                        # select image to replace leads with
                        leads_to_replace = np.random.choice([0, 1, 2, 3], size=np.random.randint(1, 2+1), replace=False)
                        for lead_to_replace in leads_to_replace:
                            channel = '2' if 'channel=2' in path else '1' if 'channel=1' in path else '0'
                            key = f'{sample_class}_{channel}'
                            get_superimpose_path = lambda: images_by_class_and_channel[key][np.random.randint(0, len(images_by_class_and_channel[key]))]
                            superimpose_path = get_superimpose_path()
                            while os.path.samefile(superimpose_path, file_path):
                                superimpose_path = get_superimpose_path()
                            with Image.open(superimpose_path) as superimpose_img:
                                superimpose_lead = superimpose_img.crop((0,
                                                                         lead_to_replace * CLASSIFICATION_NETWORK_LEAD_HEIGHT,
                                                                         superimpose_img.width-1,
                                                                         (lead_to_replace + 1) * CLASSIFICATION_NETWORK_LEAD_HEIGHT))
                                orig_draw.rectangle((0,
                                                     lead_to_replace * CLASSIFICATION_NETWORK_LEAD_HEIGHT,
                                                     orig_img.width-1,
                                                     (lead_to_replace + 1) * CLASSIFICATION_NETWORK_LEAD_HEIGHT), fill='#000000')
                                orig_img.paste(superimpose_lead, (0, lead_to_replace * CLASSIFICATION_NETWORK_LEAD_HEIGHT))
                        
                        for lead_to_stretch in [0, 1, 2, 3]:
                            superimpose_lead = orig_img.crop((0, lead_to_stretch * CLASSIFICATION_NETWORK_LEAD_HEIGHT,
                                                              orig_img.width - 1,
                                                              (lead_to_stretch + 1) * CLASSIFICATION_NETWORK_LEAD_HEIGHT))
                            superimpose_lead = superimpose_lead.resize((int(superimpose_lead.width * max(0.5, np.random.normal(1.0, 0.1))),
                                                                        int(superimpose_lead.height * max(0.5, np.random.normal(1.0, 0.1)))))
                            superimpose_lead.thumbnail((CLASSIFICATION_NETWORK_IMG_WIDTH, CLASSIFICATION_NETWORK_LEAD_HEIGHT),
                                                       Image.ANTIALIAS)
                            orig_draw.rectangle((0,
                                                 lead_to_stretch * CLASSIFICATION_NETWORK_LEAD_HEIGHT,
                                                 orig_img.width-1,
                                                 (lead_to_stretch + 1) * CLASSIFICATION_NETWORK_LEAD_HEIGHT), fill='#000000')
                            orig_img.paste(superimpose_lead, (0, lead_to_stretch * CLASSIFICATION_NETWORK_LEAD_HEIGHT))
                            
                        orig_img.save(os.path.join(ptb_v_classification_aug_dir, str(next_aug_idx) + '_'.join(os.path.basename(path).split('_')[1:]) + f'_aug{path_ext}'))
                next_aug_idx += 1

                # pick any other image from 

        except Exception as e:
            continue
