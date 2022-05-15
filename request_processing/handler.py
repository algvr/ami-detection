from http.server import  BaseHTTPRequestHandler, HTTPServer

import argparse
import base64
from io import BytesIO
import json
import math
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image, ImageDraw, ImageFont
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as T

import data_handling.dataloader
from ecg_plotting import DEFAULT_PAPER_SPEED, DEFAULT_SMALL_CELL_SIZE, DEFAULT_TIMESTEPS_PER_SECOND
from datasets.ptb_xl.data_handling import get_ecg_array, PTB_XL_LEAD_LABELS
from .hough_processing import *
from utils import *

from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.data.all import *
from fastai.vision.all import *

from fastai.callback.core import *

import fastai.vision.core

# from fastai.vision.widgets import *

def get_new_point_pos(pt_x, pt_y, angle_deg, img_width, img_height):
    # credit to https://stackoverflow.com/a/51964802
    center_x = int(img_width / 2)
    center_y = int(img_height / 2)

    angle_rad = (angle_deg / 180.0) * math.pi
    new_px = center_x + int(float(pt_x - center_x) * math.cos(angle_rad)
                            + float(pt_y - center_y) * math.sin(angle_rad))
    new_py = center_y + int(-(float(pt_x - center_x) * math.sin(angle_rad))
                            + float(pt_y - center_y) * math.cos(angle_rad))
    return new_px, new_py


class CustomHTTPRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, request, client_addr, server):
        super(CustomHTTPRequestHandler, self).__init__(request, client_addr, server)

        # set model here
        self.segmentation_model = UNetTF
        
        
        # !!! set checkpoint of segmentation model to load here !!!


        load_checkpoint_path = os.path.join('downloaded_cps', 'cp_ep-00010_it-00710_step-10400.ckpt')
        # change depending on whether this is a TF model or a Torch model
        self.segmentation_model.load_weights(load_checkpoint_path)

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        
    def do_OPTIONS(self):
        # credit to https://stackoverflow.com/a/32501309
        self.send_response(200, 'ok')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get('content-length'))
        field_data = self.rfile.read(length)
        fields = json.loads(field_data) 

        if 'test' in fields:
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'{"success": true}')
        elif not 'objects' in fields or not 'image_str' in fields or not 'token' in fields:
            self.send_response(400)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
        else:
            objects = fields['objects']
            image_str = fields['image_str']
            token = fields['token']
            try:
                self.send_response(200)
                result = process_request(self.segmentation_model, objects, image_str, token)
                result_str = json.dumps(result)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(str.encode(result_str))
            except Exception as e:
                print(str(e))
                self.send_response(400)
                self.end_headers()


def process_request(segmentation_model, objects, image_str, token, original_sample_json_path=None,
                    generate_time_series=False, learner=None, classification_model_name=None):
    if original_sample_json_path not in ['', None]:
        ecg_dataset_X, ecg_dataset_Y = get_ecg_array()
    else:
        ecg_dataset_X, ecg_dataset_Y = None, None

    log_dict = {}

    log_dict['request_objects'] = objects
    log_dict['request_token'] = token

    comma_idx = image_str.find(',')
    if comma_idx > -1:
        image_str = image_str[comma_idx+1:]

    image_str = image_str.strip()

    decoded_obj = base64.b64decode(image_str)
    bytesio_obj = BytesIO(decoded_obj)
    img = Image.open(bytesio_obj).convert('RGB')
    img_width, img_height = img.size

    found_lead_names = []

    # extract leads
    extracted_lead_pre_rot_imgs = {}
    num_leads_processed = 0
    for idx, v in enumerate(objects.values()):
        if 'type' not in v or v['type'] != 'rect' or 'lineCoords' not in v:
            continue
        
        tl_x, tl_y = v['lineCoords']['tl']['x'], v['lineCoords']['tl']['y']
        tr_x, tr_y = v['lineCoords']['tr']['x'], v['lineCoords']['tr']['y']
        
        bl_x, bl_y = v['lineCoords']['bl']['x'], v['lineCoords']['bl']['y']
        br_x, br_y = v['lineCoords']['br']['x'], v['lineCoords']['br']['y']

        # note that the rotation may change the relative order of the points

        angle = v['angle']
        rot_kwargs = {'angle_deg': angle, 'img_width': img_width, 'img_height': img_height}
        tl_x_rot, tl_y_rot = get_new_point_pos(tl_x, tl_y, **rot_kwargs)
        tr_x_rot, tr_y_rot = get_new_point_pos(tr_x, tr_y, **rot_kwargs)
        bl_x_rot, bl_y_rot = get_new_point_pos(bl_x, bl_y, **rot_kwargs)
        br_x_rot, br_y_rot = get_new_point_pos(br_x, br_y, **rot_kwargs)

        leftmost_x   = min([tl_x_rot, tr_x_rot, bl_x_rot, br_x_rot])
        topmost_y    = min([tl_y_rot, tr_y_rot, bl_y_rot, br_y_rot])
        
        rightmost_x  = max([tl_x_rot, tr_x_rot, bl_x_rot, br_x_rot])
        bottommost_y = max([tl_y_rot, tr_y_rot, bl_y_rot, br_y_rot])

        rot_img = img.rotate(angle)

        crop_rot_img = rot_img.crop((leftmost_x, topmost_y, rightmost_x, bottommost_y))

        lead_name = v['auxText'].split()[-1].upper()

        found_lead_names.append(lead_name)

        extracted_lead_pre_rot_imgs[lead_name] = crop_rot_img

        num_leads_processed += 1

    # find valid resolutions for segmentation model
    segmentation_model_name = segmentation_model.name if hasattr(segmentation_model, 'name')\
                              else type(segmentation_model).__name__
    segmentation_model_valid_res_file_path = os.path.join('valid_res_lists', f'{segmentation_model_name}.json')
    if not os.path.exists(segmentation_model_valid_res_file_path):
        raise RuntimeError(f'Valid resolution list for model "{segmentation_model_name}" not found '
                            f'(looked for "{segmentation_model_valid_res_file_path}").')
    with open(segmentation_model_valid_res_file_path, 'r') as f:
        config_data = json.loads(f.read())
        segmentation_model_valid_resolutions = config_data['valid_resolutions']

    # generate segmentations
    extracted_lead_pre_rot_segmentations = {}
    for lead_name, lead_img in extracted_lead_pre_rot_imgs.items():
        # resize image to valid resolution
        lead_img_width, lead_img_height = lead_img.size
        final_width, final_height =\
            data_handling.dataloader.DataLoader._get_valid_sample_resolution(lead_img_width, lead_img_height, segmentation_model_valid_resolutions)

        log_dict[f'original_resolution_{lead_name}_width'] = lead_img_width
        log_dict[f'original_resolution_{lead_name}_height'] = lead_img_height
        
        log_dict[f'valid_resolution_{lead_name}_width'] = final_width
        log_dict[f'valid_resolution_{lead_name}_height'] = final_height
        
        # (0, 0, 0) is also used during training
        final_seg_input_img = Image.new('RGB', (final_width, final_height), (0, 0, 0))
        final_seg_input_img.paste(lead_img, (0, 0))  # paste to top-left corner

        final_seg_input_np = np.expand_dims(np.array(final_seg_input_img), 0).astype(np.float32) / 255.0

        # we don't need to log this
        extracted_lead_pre_rot_segmentations[lead_name] = segmentation_model.predict(final_seg_input_np)
        extracted_lead_pre_rot_segmentation_img =\
            Image.fromarray(np.squeeze(extracted_lead_pre_rot_segmentations[lead_name] / 3.0 * 255.0).astype(np.uint8))
        
        log_dict[f'extracted_lead_pre_rot_segmentation_{lead_name}'] = extracted_lead_pre_rot_segmentation_img
    pass

    # straighten image
    straightened_seg_imgs = {}
    lead_deg_rot_angles = {}
    mean_hor_dists, median_hor_dists, mean_vert_dists, median_vert_dists = {}, {}, {}, {}
    for lead_name, seg in extracted_lead_pre_rot_segmentations.items():
        # filter only the horizontal and vertical lines
        seg_argmax = np.argmax(seg, axis=-1)
        seg_img = np.squeeze(np.logical_or(seg_argmax == 1, seg_argmax == 2) * 255)
        seg_img_differentiated = np.squeeze((seg_argmax.astype(np.float32) / 3.0 * 255.0).astype(np.uint8))

        deg_rot_angle = get_deg_rot_angle_loop(seg_img_differentiated.astype(np.uint8), log_dict, lead_name)
        lead_deg_rot_angles[lead_name] = deg_rot_angle

        log_dict[f'deg_rot_angle_{lead_name}'] = deg_rot_angle


        # use the original selection, not the segmentation

        str_img = Image.fromarray(seg_img.astype(np.uint8)).rotate(angle=deg_rot_angle, expand=False)
        straightened_seg_imgs[lead_name] = str_img

        str_img_differentiated = Image.fromarray(seg_img_differentiated.astype(np.uint8)).rotate(angle=deg_rot_angle,
                                                                                                 expand=False)
        
        log_dict[f'straightened_orig_img_{lead_name}'] =\
            extracted_lead_pre_rot_imgs[lead_name].rotate(angle=deg_rot_angle, expand=False)

        # will be very similar to str_img; no need to log both
        log_dict[f'straightened_seg_img_{lead_name}'] = str_img_differentiated
        
        mean_hor_dist, median_hor_dist, mean_vert_dist, median_vert_dist =\
            detect_hor_vert_cell_distances_loop(pillow_to_cv2(str_img_differentiated), log_dict, lead_name)
        
        mean_hor_dists[lead_name] = mean_hor_dist
        median_hor_dists[lead_name] = median_hor_dist
        mean_vert_dists[lead_name] = mean_vert_dist
        median_vert_dists[lead_name] = median_vert_dist

        log_dict[f'mean_hor_dist_{lead_name}'] = mean_hor_dist
        log_dict[f'median_hor_dist_{lead_name}'] = median_hor_dist
        log_dict[f'mean_vert_dist_{lead_name}'] = mean_vert_dist
        log_dict[f'median_vert_dist_{lead_name}'] = median_vert_dist

    # generating the segmentations again is too computationally expensive. instead, rotate the image
    # and use nearest neighbor interpolation (or similar interpolation type that doesn't produce intermediate values,
    # or could even use intermediate-value-producing interpolation type, but then perform rounding)

    # generate segmentations again
    #extracted_lead_post_rot_segmentations = {}
    #for lead_name, lead_img in straightened_lead_imgs.items():
    #    # resize image to valid resolution
    #    lead_img_width, lead_img_height = lead_img.size
    #    final_width, final_height =\
    #        data_handling.dataloader.DataLoader._get_valid_sample_resolution(lead_img_width, lead_img_height, valid_resolutions)
    #        
    #    final_seg_input = Image.new('RGB', (final_width, final_height), (0, 0, 0))
    #    final_seg_input.paste(lead_img, (0, 0))  # paste to top-left corner
    #    
    #    extracted_lead_post_rot_segmentations[lead_name] = model.predict(np.expand_dims(np.array(final_seg_input), 0))

    # rotate segmentations
    
    largest_post_rot_seg_img_width = 0

    extracted_lead_post_rot_segmentations = {}
    for lead_name, seg in extracted_lead_pre_rot_segmentations.items():
        seg_argmax_squeezed = np.squeeze(np.argmax(seg, axis=-1).astype(np.uint8))
        seg_img = Image.fromarray(seg_argmax_squeezed)  # if interpreted as literal image, will be very dark
        rot_img = seg_img.rotate(lead_deg_rot_angles[lead_name], resample=Image.NEAREST)
        post_rot_seg = np.array(rot_img).astype(np.uint8)
        extracted_lead_post_rot_segmentations[lead_name] = post_rot_seg

        post_rot_seg_img = Image.fromarray((post_rot_seg.astype(np.float32) / 3.0) * 255.0).convert('RGB')
        if post_rot_seg_img.width > largest_post_rot_seg_img_width:
            largest_post_rot_seg_img_width = post_rot_seg_img.width
        log_dict[f'post_rot_seg_img_{lead_name}'] = post_rot_seg_img


    longest_residual_time_series_length = 0

    # extract ECG curves

    extracted_ecg_residual_time_series = {}
    for lead_name, seg in extracted_lead_post_rot_segmentations.items():
        # DBSCAN clustering on pixels seems to be too expensive
        # also note that if we partition the image into rectangles, we may have only rectangles with dark line elements,
        # so using k-Means with k=2 is probably not a good idea
        
        # maybe can apply mean shift? -> no: we would lose detail

        # good idea: DBSCAN on *proportion of ECG curve pixels*, then select leftmost and rightmost as corners
        # note that we should use the *proportion* here, not the actual number of pixels, because the outermost elements
        # may have less pixels overall

        # cheaper idea: discard where number of black pixels is more than 1 s.d. BELOW mean
        # bad idea: distribution may be unimodal -> would eliminate a lot of good parts

        # even the minimum part can have a large enough amount of black pixels if all parts contain an equal amount of
        # the curve

        # NOTE: a partitioning approach may cut off the corners of the ECG, but hopefully not too much info will be 
        # lost

        # problem with DBSCAN: takes eps and minPts (min # of points to form dense region)
        # -> if minPts = 1, even blank parts will form a region

        # another idea: take maximum proportion (NOT mean); do not allow proportion of black pixels to fall below 50%
        # of maximum
        # maximum and not mean because most regions might be blank! 

        def get_sample_data():
            sample_idx, img_fn, is_normal, is_non_mi_abnormality, is_mi = None, None, None, None, None
            if original_sample_json_path not in ['', None]:
                try:
                    with open(original_sample_json_path) as f:
                        config_json = json.loads(f.read())
                        is_mi = config_json['_is_mi']
                        is_non_mi_abnormality = config_json['_is_non_mi_abnormality']
                        is_normal = config_json['_is_normal']
                        img_fn = os.path.basename(config_json['_img_path'])
                        img_fn_components = img_fn.split('_')
                        # try to find corresponding recording
                        try:
                            sample_idx = int(img_fn_components[0])
                        except:
                            sample_idx = None
                except:
                    pass
            return sample_idx, img_fn, is_normal, is_non_mi_abnormality, is_mi

        sample_idx, sample_img_fn, sample_is_normal, sample_is_non_mi_abnormality, sample_is_mi = get_sample_data()
        sample_lbl = next(filter(lambda e: e[0], [[sample_is_normal, 'normal'],
                                                  [sample_is_non_mi_abnormality, 'non_mi_abnormality'],
                                                  [sample_is_mi, 'mi'],
                                                  [True, None]]))[1]


        if False and sample_idx is not None and classification_model_name is not None:
            # quit if already processed

            ptb_v_classification_dir = os.path.join('datasets', 'ptb_v_classification', 'training')
            os.makedirs(ptb_v_classification_dir, exist_ok=True)

            prefix = str(sample_idx) + '__' + sample_lbl + '__'
            for filename in os.listdir(ptb_v_classification_dir):
                if filename.startswith(prefix):
                    return


        # question: how to handle parts left out in ECG -> just impute with previous value?

        # partition image into 50 pieces


        # TODO: test this!

        ecg_curve_seg = (seg == 3).astype(np.float32)

        seg_height, seg_width = seg.shape
        
        log_dict[f'seg_height_{lead_name}'] = seg_height
        log_dict[f'seg_width_{lead_name}'] = seg_width

        NUM_PARTITIONS = min(50, seg_width)  # may not be actual number due to division; just a guideline
        part_ecg_proportions = []

        x_step = max(1, int(seg_width / NUM_PARTITIONS))
        for part_x_start in range(0, seg_width, x_step):
            seg_part = ecg_curve_seg[:, part_x_start:(part_x_start + x_step)]
            part_ecg_proportion = ecg_curve_seg.sum() / (seg_part.shape[0] * seg_part.shape[1])
            part_ecg_proportions.append(part_ecg_proportion)

        part_ecg_proportions_np = np.array(part_ecg_proportions)
        
        part_ecg_proportions_max = np.mean(part_ecg_proportions_np)

        log_dict[f'part_ecg_proportions_max_{lead_name}'] = part_ecg_proportions_max

        first_ecg_containing_part_ltr = -1
        first_ecg_containing_part_rtl = -1

        MIN_ECG_PROPORTION_COMPARED_TO_MAX = 0.5

        num_parts = len(part_ecg_proportions)
        log_dict[f'num_parts_{lead_name}'] = num_parts

        for part_x_idx in range(num_parts):
            if part_ecg_proportions[part_x_idx] / part_ecg_proportions_max >= MIN_ECG_PROPORTION_COMPARED_TO_MAX:
                first_ecg_containing_part_ltr = part_x_idx
                break
        
        for part_x_idx in range(num_parts - 1, -1, -1):
            if part_ecg_proportions[part_x_idx] / part_ecg_proportions_max >= MIN_ECG_PROPORTION_COMPARED_TO_MAX:
                first_ecg_containing_part_rtl = part_x_idx
                break

        if first_ecg_containing_part_ltr > first_ecg_containing_part_rtl:  # shouldn't happen
            first_ecg_containing_part_rtl = first_ecg_containing_part_ltr

        # NOTE: we chould do the same with the y dimension! cut out unnecessary blocks
        # however, this could be nontrivial... so let's just use the Y coordinate extraction approach below:
        
        # we could simply create an array with image Y coordinates, multiply with the segmentation mask,
        # then take the mean across the height dimension (will likely have to flip H and W channels)
        # will have to sum up the coordinates of each row, then divide by the number of nonzero elements in each row

        ecg_curve_seg_sliced =\
            ecg_curve_seg[:, (first_ecg_containing_part_ltr * x_step):((first_ecg_containing_part_rtl + 1) * x_step)]

        y_column = np.arange(0, seg_height)
        y_coord_img = np.moveaxis(np.repeat(np.expand_dims(y_column, 0), ecg_curve_seg_sliced.shape[1], axis=0), 0, 1)
         
        # note that we can use neither the mean nor the median here, since we would most likely always get values close
        # to 0
        y_coord_sum = np.sum(ecg_curve_seg_sliced * y_coord_img, axis=0)
        column_nonzero_counts = np.sum(ecg_curve_seg_sliced, axis=0)

        column_nonzero_counts_nonzeros = np.nonzero(column_nonzero_counts)

        x_coords_filtered = np.arange(first_ecg_containing_part_ltr * x_step,
                                      (first_ecg_containing_part_rtl + 1) * x_step)[column_nonzero_counts_nonzeros]
        y_coord_sum_filtered = y_coord_sum[column_nonzero_counts_nonzeros]
        column_nonzero_counts_filtered = column_nonzero_counts[column_nonzero_counts_nonzeros]

        # construct a Pandas Series objects from the points we have

        # we can use median_vert_dists[lead_name] and median_hor_dists[lead_name] here

        # if a NaN is detected, just choose the median value 

        x_pixels_per_large_cell = median_hor_dists[lead_name]
        y_pixels_per_large_cell = median_vert_dists[lead_name]

        if np.isnan(x_pixels_per_large_cell):
            x_pixels_per_large_cell = np.nanmedian(list(median_hor_dists.values()))
            
        if np.isnan(y_pixels_per_large_cell):
            y_pixels_per_large_cell = np.nanmedian(list(median_vert_dists.values()))

        log_dict[f'x_pixels_per_large_cell_{lead_name}'] = x_pixels_per_large_cell
        log_dict[f'y_pixels_per_large_cell_{lead_name}'] = y_pixels_per_large_cell

        # paper speed: e.g. 25mm/1s

        # one small cell always measures 1mm x 1mm -> one large cell always measures 5mm x 5mm
        # (1.0 / paper_speed) gives s/mm -> seconds per millimeter
        # (1.0 / paper_speed) * 5mm gives seconds per large cell
        
        paper_speed = DEFAULT_PAPER_SPEED
        seconds_per_large_cell = (1.0 / paper_speed) * 5.0

        # TODO: generalize to arbitrary paper speeds!
        # note that because of the way we process the time series (scanning instead of feeding the network the raw data),
        # we will be able to deal with arbitrary paper speeds, and do not need to retrain the network!
        # this is a benefit we should mention

        # determine seconds per pixel
        seconds_per_pixel = seconds_per_large_cell / x_pixels_per_large_cell

        log_dict[f'seconds_per_pixel_{lead_name}'] = seconds_per_pixel

        # further ideas:
        # plot y_coord_sum_filtered
        # plot series
        # plot interpolated
        # make it easy to enable/disable plotting

        # plot y_coord_sum_filtered

        def plot_y_coord_sum_filtered():
            y_coord_sum_filtered_img = Image.new('RGBA', (seg_width, seg_height), (0, 0, 0, 0))
            for x_coord_idx, x_coord in enumerate(x_coords_filtered):
                y_coord_sum_filtered_img.putpixel((x_coord,
                                                   int(y_coord_sum_filtered[x_coord_idx] /
                                                       column_nonzero_counts_filtered[x_coord_idx])),
                                                  (255, 0, 0, 255))

            log_dict[f'y_coord_sum_filtered_img_{lead_name}'] = y_coord_sum_filtered_img

        plot_y_coord_sum_filtered()

        series = pd.Series(((y_coord_sum_filtered / column_nonzero_counts_filtered) / y_pixels_per_large_cell) * seconds_per_large_cell,
                            index=pd.TimedeltaIndex(['%.06fs' % s for s in (seconds_per_pixel * x_coords_filtered)]))

        freq_str = '%.06fS' % (1.0 / CLASSIFICATION_RNN_TIMESTEPS_PER_SECOND)

        series_resampled_interpolated = series.resample(freq_str).mean().interpolate(method='linear')

        # next step: resample to get some frequency (e.g. 100Hz -> 0.01s), then interpolate, then compute residual,
        # then feed to network


        # TODO: try other interpolation methods, e.g. splines

        def plot_series_resampled_interpolated():
            series_resampled_interpolated_img = Image.new('RGBA', (LOGGING_IMG_WIDTH, seg_height), (0, 0, 0, 0))
            for x_coord, pre_y_coord in enumerate(series_resampled_interpolated):
                y_coord = int(pre_y_coord * y_pixels_per_large_cell / seconds_per_large_cell)
                if x_coord < LOGGING_IMG_WIDTH and y_coord < seg_height:
                    series_resampled_interpolated_img.putpixel((x_coord, y_coord), (255, 0, 0, 255))
            log_dict[f'series_resampled_interpolated_img_{lead_name}'] = series_resampled_interpolated_img

        plot_series_resampled_interpolated()

        pass

        def plot_original_series():
            if sample_idx is not None:
                lead_idx = next((lead_idx for lead_idx, lead_lbl in PTB_XL_LEAD_LABELS
                                 if lead_lbl.upper() == lead_name.upper()), -1)
                if lead_idx > -1:
                    original_ecg_series = ecg_dataset_X[sample_idx, lead_idx]
                    original_ecg_series_img = Image.new('RGBA', (seg_width, seg_height), (0, 0, 0, 0))
                    # original series will most definitely extend beyond the input, since we only plot
                    # the first ~2s of each lead
                    for x_coord, pre_y_coord in enumerate(original_ecg_series):
                        y_coord = int(pre_y_coord * DEFAULT_SMALL_CELL_SIZE * 5)
                        if x_coord < seg_width and y_coord < seg_height:
                            original_ecg_series_img.putpixel((x_coord, y_coord), (255, 0, 0, 255))
                    log_dict[f'original_ecg_series_img_{lead_name}'] = original_ecg_series_img

        # now, take residual
        # note that the negative direction corresponds to "up"!

        series_resampled_interpolated_np = np.array(series_resampled_interpolated)
        residual_time_series = series_resampled_interpolated_np[1:] - series_resampled_interpolated_np[:-1]
        
        extracted_ecg_residual_time_series[lead_name] = residual_time_series

        if residual_time_series.shape[0] > longest_residual_time_series_length:
            longest_residual_time_series_length = residual_time_series.shape[0]

    # now, create a uniform-length array of the leads, arrange them in the standard layout, then perform classification

    rnn_input_leads = []

    for current_lead_name in PTB_XL_LEAD_LABELS:
        if current_lead_name.split()[-1].upper() in extracted_ecg_residual_time_series:
            lead_time_series = extracted_ecg_residual_time_series[lead_name]
            diff_to_max_length = longest_residual_time_series_length - lead_time_series.shape[0]
            if diff_to_max_length > 0:
                lead_time_series_padded = np.concatenate((lead_time_series, np.zeros(diff_to_max_length)), axis=0)
            else:
                lead_time_series_padded = lead_time_series
            time_series_to_append = lead_time_series_padded
        else:
            time_series_to_append = np.zeros(longest_residual_time_series_length)
        
        rnn_input_leads.append(time_series_to_append)

    rnn_input = np.stack(rnn_input_leads)
    
    # pickle the time series:

    if sample_idx is not None and generate_time_series:
        ptb_reconstructed_dir = os.path.join('datasets', 'ptb_reconstructed', 'training')
        os.makedirs(ptb_reconstructed_dir, exist_ok=True)
        sample_time_series_path =\
            os.path.join(ptb_reconstructed_dir,
                         str(sample_idx) + '__' + sample_lbl + '__' + os.path.basename(sample_img_fn) + '.dat')
        with open(sample_time_series_path, 'wb') as f:
            pickle.dump(rnn_input, f)

    # generate the image-based classification input

    if sample_idx is not None and classification_model_name is not None:
        # determine valid resolution

        ptb_v_classification_dir = os.path.join('datasets', 'ptb_v_classification', 'training')
        os.makedirs(ptb_v_classification_dir, exist_ok=True)

        # find valid resolutions for classification model
        classification_model_valid_res_file_path = os.path.join('valid_res_lists', f'{classification_model_name}.json')
        if not os.path.exists(classification_model_valid_res_file_path):
            raise RuntimeError(f'Valid resolution list for model "{classification_model_name}" not found '
                                f'(looked for "{classification_model_valid_res_file_path}").')
        with open(classification_model_valid_res_file_path, 'r') as f:
            config_data = json.loads(f.read())
            classification_model_valid_resolutions = config_data['valid_resolutions']

        # not largest_post_rot_seg_img_width (need fixed size to use FC layers)
        classification_input_img_width_per_channel, classification_input_img_height_per_channel =\
            data_handling.dataloader.DataLoader._get_valid_sample_resolution(CLASSIFICATION_NETWORK_MAX_LEAD_WIDTH,
                                                                             4 * CLASSIFICATION_NETWORK_LEAD_HEIGHT,
                                                                             classification_model_valid_resolutions)


        # since the ResNet has 3 input channels, distribute the leads across the channels
        classification_input_imgs = [Image.new('L', (classification_input_img_width_per_channel,
                                                     classification_input_img_height_per_channel), 0)
                                     for _ in range(3)]

        classification_merged_input_img = Image.new('L', (640, 640), 0)
        
        # initialize with all white (may want to change this to check if doing so proves more beneficial)
        for current_lead_idx, current_lead_name in enumerate(PTB_XL_LEAD_LABELS):
            classification_input_img = classification_input_imgs[int(current_lead_idx / 4)]
            lead_name_normalized = current_lead_name.split()[-1].upper()
            img_key_name = f'post_rot_seg_img_{lead_name_normalized}'
            if img_key_name in log_dict:
                # as for now, do not stretch, but do condense
                img_to_paste = log_dict[img_key_name].convert('L')
                # in-place operation:
                img_to_paste.thumbnail((CLASSIFICATION_NETWORK_MAX_LEAD_WIDTH, CLASSIFICATION_NETWORK_LEAD_HEIGHT),
                                       resample=Image.BICUBIC)  # NEAREST resampling: lots of curve details lost!
                classification_input_img.paste(img_to_paste,
                                               (0, (current_lead_idx % 4) * CLASSIFICATION_NETWORK_LEAD_HEIGHT))

                img_to_paste.thumbnail((210, CLASSIFICATION_NETWORK_LEAD_HEIGHT), resample=Image.BICUBIC)
                classification_merged_input_img.paste(img_to_paste, (int(current_lead_idx / 4) * 210,
                                                      (current_lead_idx % 4) * CLASSIFICATION_NETWORK_LEAD_HEIGHT))

        sample_classification_input_img_paths = []
        for channel_idx in range(3):
            classification_input_img = classification_input_imgs[channel_idx]
            sample_classification_input_img_path =\
                os.path.join(ptb_v_classification_dir,
                            str(sample_idx) + '__' + sample_lbl + '__' + os.path.basename(sample_img_fn)
                            + f'__channel={channel_idx}' + '.png')
            classification_input_img.save(sample_classification_input_img_path)
            sample_classification_input_img_paths.append(sample_classification_input_img_path)

        if original_sample_json_path not in ['', None]:
            with open(original_sample_json_path, 'r+') as f:
                config = json.loads(f.read())
                config['_classification_input_img_paths'] = sample_classification_input_img_paths
                f.seek(0)
                f.write(json.dumps(config))
                f.truncate()

    # next step: create comprehensive visualization

    # dict_keys(['request_objects', 'request_token', 'original_resolution_I_width', 'original_resolution_I_height', 'valid_resolution_I_width', 'valid_resolution_I_height', 'extracted_lead_pre_rot_segmentations_I', 'original_resolution_AVR_width', 'original_resolution_AVR_height', 'valid_resolution_AVR_width', 'valid_resolution_AVR_height', 'extracted_lead_pre_rot_segmentations_AVR', 'lines_pre_rot', 'deg_rot_angle_I', 'straightened_lead_img_I', 'cluster_lines_rec_post_rot', 'mean_hor_dists_I', 'median_hor_dists_I', 'mean_vert_dists_I', 'median_vert_dists_I', 'hough_transform_deg_rot_angle_AVR', 'straightened_lead_img_AVR', 'mean_hor_dists_AVR', 'median_hor_dists_AVR', 'mean_vert_dists_AVR', 'median_vert_dists_AVR', 'post_rot_seg_img_I', 'post_rot_seg_img_AVR', 'seg_height_I', 'seg_width_I', 'part_ecg_proportions_max_I', 'num_parts_I', 'x_pixels_per_large_cell_I', 'y_pixels_per_large_cell_I', 'seconds_per_pixel_I', 'y_coord_sum_filtered_img_I', 'series_resampled_interpolated_I', 'seg_height_AVR', 'seg_width_AVR', 'part_ecg_proportions_max_AVR', 'num_parts_AVR', 'x_pixels_per_large_cell_AVR', 'y_pixels_per_large_cell_AVR', 'seconds_per_pixel_AVR', 'y_coord_sum_filtered_img_AVR', 'series_resampled_interpolated_AVR'])

    # layout: 

    # for each lead (could simply write function that generates image for one lead; maybe better, to be able to isolate):

    # big title: {lead_name}
    # original_resolution_width x original_resolution_height: (values)
    # valid_resolution_width x valid_resolution_height: (values)
    # deg_rot_angle: (value)
    # mean_hor_dist: (value); median_hor_dist: (value)
    # mean_vert_dist: (value); median_vert_dist: (value)
    # seg_width: (value); seg_height: (value)
    # part_ecg_proportions_max: (value)
    # num_parts: (value)
    # x_pixels_per_large_cell: (value); y_pixels_per_large_cell: (value)
    # seconds_per_pixel
    # lines_pre_rot:
    # (image)
    # extracted_lead_pre_rot_segmentation:
    # (image)
    # straightened_lead_img:
    # (image)
    # cluster_lines_rec_post_rot:
    # (image)
    # post_rot_seg_img
    # (image)
    # y_coord_sum_filtered_img:
    # (image)
    # series_resampled_interpolated_img:
    # (image)

    # TODO: also add Hough transform parameters!

    lead_logging_font = ImageFont.truetype('dataset_generation/fonts/courier_prime_bold.ttf', size=18)

    def get_lead_details_img(lead_name):
        # just use max width of 1000px, max height of 2600px 
        # cut off everything exceeding that boundary
        img = Image.new('RGBA', (LOGGING_IMG_WIDTH, LOGGING_IMG_HEIGHT), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        lead_log_rows = []  # may contain strings *or* PIL images
        
        for text_param_name in ['original_resolution_width', 'original_resolution_height', 'valid_resolution_width', 'valid_resolution_height', 'deg_rot_angle', 'mean_hor_dist', 'median_hor_dist',
                                'mean_vert_dist', 'median_vert_dist', 'seg_width', 'seg_height', 'x_pixels_per_large_cell', 'part_ecg_proportions_max', 'x_pixels_per_large_cell', 'y_pixels_per_large_cell']:
            if f'{text_param_name}_{lead_name}' in log_dict:
                value = log_dict[f"{text_param_name}_{lead_name}"]
                value_str = '%.04f' % value if isinstance(value, float) else str(value)
                lead_log_rows.append(f'{text_param_name}_{lead_name}: {value_str}')
        
        for img_param_name in ['lines_pre_rot', 'extracted_lead_pre_rot_segmentation', 'straightened_orig_img', 'straightened_seg_img', 'lines_post_rot', 'lines_rec_mean_shift', 'lines_hor_vert',
                               'post_rot_seg_img', 'y_coord_sum_filtered_img', 'series_resampled_interpolated_img', 'original_ecg_series_img', 'classification_merged_input_img']:
            if f'{img_param_name}_{lead_name}' in log_dict:
                lead_log_rows.append(f'{img_param_name}_{lead_name}:')
                # maybe additionally resize the image
                lead_log_rows.append(log_dict[f'{img_param_name}_{lead_name}'])
        
        y_position = 0
        MARGIN_BETWEEN_LOG_ROWS = 10
        for log_row in lead_log_rows:
            if isinstance(log_row, Image.Image):
                log_row_rgba = log_row.convert('RGBA')
                img.paste(log_row_rgba, (0, y_position), log_row_rgba)
                y_position += log_row.height + MARGIN_BETWEEN_LOG_ROWS  # add some margin
            elif isinstance(log_row, str):
                text_width, text_height = draw.textsize(log_row, font=lead_logging_font)
                draw.text((0, y_position), log_row, fill='#000000', align='left', font=lead_logging_font)
                y_position += text_height + MARGIN_BETWEEN_LOG_ROWS
    
        return img
    
    # create visualizations

    timestamp = int(time.time())

    checkpoint_path = getattr(segmentation_model, '_checkpoint_path', '')
    seg_dir = os.path.join('visualizations', checkpoint_path if checkpoint_path != '' else segmentation_model.__name__,
                           str(timestamp))
    if seg_dir != '':
        os.makedirs(seg_dir, exist_ok=True)

    for lead_name in found_lead_names:
        img = get_lead_details_img(lead_name)
        img.convert('RGB').save(os.path.join(seg_dir, f'{lead_name}.jpg'))
        
    # classification_merged_input_img.save(os.path.join(seg_dir, f'classification_merged_input_img.jpg'))


    if sample_idx is not None:
        print(f'Processed sample #{sample_idx}')
    
    #img_tensor = torch.tensor(np.moveaxis(np.array(classification_merged_input_img.convert('RGB')), -1, 0).astype(np.float32)).unsqueeze(dim=0)


    #img_tensor = T.ToTensor()(classification_merged_input_img.convert('RGB'))
    #img_tensor = torch.tensor(np.array(classification_merged_input_img.convert('RGB'))).cpu()
    #imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    #img_tensor[0, 0] = ((img_tensor[0, 0] / 255.0) - imagenet_stats[0][0]) / imagenet_stats[1][0]
    #img_tensor[0, 1] = ((img_tensor[0, 1] / 255.0) - imagenet_stats[0][1]) / imagenet_stats[1][1]
    #img_tensor[0, 2] = ((img_tensor[0, 2] / 255.0) - imagenet_stats[0][2]) / imagenet_stats[1][2]

    if learner is not None:
        pred_input = fastai.vision.core.PILImage.create(np.array(classification_merged_input_img.convert('RGB')))

        pred = learner.predict(pred_input)[2]
        print('')
        print(f'>>>>>>>>>> Prediction: {pred} <<<<<<<<<<')
        print('')

        return {'success': True, 'token': str(token),
                'normal': pred[0], 'non-mi-related-abnormalities': pred[1], 'mi': pred[2]}


def get_request_params_from_json(original_sample_json_path):
    # takes: path to JSON with data
    # returns: objects, image_str
    objects = {}
    with open(original_sample_json_path) as f:
        json_config = json.loads(f.read())
        for lead_idx, lead_name in enumerate(PTB_XL_LEAD_LABELS):
            if lead_name in json_config:
                tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y = [json_config[lead_name][idx] for idx in range(8)]
                lead_object_dict = {'lineCoords': {'tl': {'x': tl_x, 'y': tl_y},
                                                   'tr': {'x': tr_x, 'y': tr_y},
                                                   'bl': {'x': bl_x, 'y': bl_y},
                                                   'br': {'x': br_x, 'y': br_y}}}
                # WARNING: make sure angle is consistent with the function below (units and direction)!
                # calculate the angle (take average between top angle and bottom angle)

                angle_1 = np.arctan2(tr_x - tl_x, tr_y - tl_y) / np.pi * 180.0 - 90.0
                angle_2 = np.arctan2(br_x - bl_x, br_y - bl_y) / np.pi * 180.0 - 90.0
                lead_object_dict['angle'] = (angle_1 + angle_2) / 2.0

                lead_object_dict['auxText'] = f'Lead {lead_name}' 
                lead_object_dict['type'] = 'rect'

                objects[str(lead_idx)] = lead_object_dict

        with open(json_config['_img_path'], 'rb') as img:
            encoded_img = base64.b64encode(img.read()).decode('utf-8')
        return objects, encoded_img
