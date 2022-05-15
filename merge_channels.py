import json
import os
from PIL import Image


if __name__ == '__main__':
    ptb_v_dir = os.path.join('datasets', 'ptb_v', 'training')
    ptb_v_classification_dir = os.path.join('datasets', 'ptb_v_classification', 'training')
    ptb_v_classification_merged_dir = os.path.join('datasets', 'ptb_v_classification_merged', 'training')

    os.makedirs(ptb_v_classification_merged_dir, exist_ok=True)

    for filename in os.listdir(ptb_v_classification_dir):
        fn_root, ext = os.path.splitext(filename)
        if ext.lower() != '.png':
            continue
        fn_parts = filename.split('_')
        try:
            sample_idx = int(fn_parts[0])
            channel_idx = int(fn_root.split('=')[-1])
        except:
            continue
        
        merged_file_path = os.path.join(ptb_v_classification_merged_dir, '='.join(fn_root.split('=')[:-1])
                                        + f'=merged{ext}')
        if os.path.isfile(merged_file_path):
            continue

        imgs_to_merge = [Image.open(os.path.join(ptb_v_classification_dir, '='.join(fn_root.split('=')[:-1])
                                                 + f'={channel_idx}{ext}')) for channel_idx in range(3)]
        merged_img = Image.new('RGB', (640, 640), (0, 0, 0))
        for img_to_merge_idx, img_to_merge in enumerate(imgs_to_merge):
            merged_img.paste(img_to_merge, (img_to_merge_idx * 210, 0))

        merged_img.save(merged_file_path)

        json_path = os.path.join(ptb_v_dir, os.path.splitext(filename.split('__')[2])[0] + '_data.json')

        if os.path.isfile(json_path):
            with open(json_path, 'r+') as f:
                config = json.loads(f.read())
                config['_merged_classification_input_img_path'] = merged_file_path
                f.seek(0)
                f.write(json.dumps(config))
                f.truncate()
