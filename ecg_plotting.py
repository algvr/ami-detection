# Guglielmo Pacifico, Alexey Gavryushin
# ETH Zurich, 2022


import numpy as np
import math
import datasets.ptb_xl.data_handling as dh
from PIL import Image, ImageDraw, ImageFont


# default size of small cell on ECG image, in pixels; one small cell corresponds to 1mm
DEFAULT_SMALL_CELL_SIZE = 10

# default number of small cells (in y direction) per lead
DEFAULT_VERTICAL_SMALL_CELLS_PER_LEAD = 40

# default thick line width, for large cells encompassing 5x5 small cells
DEFAULT_THICK_CELL_LINE_WIDTH = 2

# default thin line width, for small cells
DEFAULT_THIN_CELL_LINE_WIDTH = 1

# default line width of the ECG curve
DEFAULT_CURVE_LINE_WIDTH = 3

# default paper speed, in mm/s
DEFAULT_PAPER_SPEED = 25

# default timesteps per second in an ECG recording
DEFAULT_TIMESTEPS_PER_SECOND = 100

# default millivolts per small cell
DEFAULT_MILLIVOLTS_PER_SMALL_CELL = 0.1

# default font size of lead labels (aVR, I, II, ...) and metadata (paper speed, ...) on ECG
DEFAULT_LEAD_LABEL_FONT_SIZE = 22
DEFAULT_METADATA_FONT_SIZE = 22

# default colors of thick and thin lines
THICK_LINE_COLOR = '#EC2320'
THIN_LINE_COLOR = '#ED8888'

# default order of leads, from top left to bottom right in a left-to-right manner, in a 12-lead ECG
DEFAULT_ECG_LEAD_SEQUENCE = ['I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5', 'III', 'aVF', 'V3', 'V6']

# color cycle (used for debugging purposes)
COLOR_CYCLE = ['#E69F25', '#5CB4E4', '#069F72', '#DDD040', '#0773B2', '#CC79A7', '#D35F27']


def get_ecg_image(layout_array,
                  label_array,
                  recording_array,
                  small_cell_size=DEFAULT_SMALL_CELL_SIZE,
                  vertical_small_cells_per_lead=DEFAULT_VERTICAL_SMALL_CELLS_PER_LEAD,
                  paper_speed=DEFAULT_PAPER_SPEED,
                  timesteps_per_second=DEFAULT_TIMESTEPS_PER_SECOND,
                  millivolts_per_small_cell=DEFAULT_MILLIVOLTS_PER_SMALL_CELL,
                  thick_cell_line_width=DEFAULT_THICK_CELL_LINE_WIDTH,
                  thin_cell_line_width=DEFAULT_THIN_CELL_LINE_WIDTH,
                  curve_line_width=DEFAULT_CURVE_LINE_WIDTH,
                  lead_label_font_size=DEFAULT_LEAD_LABEL_FONT_SIZE,
                  metadata_font_size=DEFAULT_METADATA_FONT_SIZE,
                  add_metadata=True):
    """
    Returns an ECG paper image from the given layout array, recording array and parameters.
    Use "get_random_ecg_photo" and "get_ecg_photo_from_image" in "data_augmentation.py" to get images of simulated ECG
    *photographs* instead.
    :param layout_array: numpy array with the shape [rows, columns, 3], specifying how to lay out the ECG
                         the last dimension contains 3 entries with [lead_idx, start_timestep, stop_timestep]
                         lead_idx describes the index of the lead (last dimension in "recording_array")
                         start_timestep and stop_timestep describe the slice of the recording at lead_idx to use
                         (first dimension in "recording_array")
    :param label_array: python list with the shape [rows, columns], specifying the label to print at each lead, with the
                        layout of the leads determined by "layout_array"
    :param recording_array: numpy array with the shape [timesteps, leads], containing the ECG recordings to use
    :param small_cell_size: size of small cell on ECG image, in pixels
    :param vertical_small_cells_per_lead: number of small cells (in y direction) per lead
    :param paper_speed: paper speed, in mm/s
    :param timesteps_per_second: timesteps per second in an ECG recording from "recording_array"
    :param millivolts_per_small_cell: millivolts per small cell
    :param thick_cell_line_width: thick line width, for large cells encompassing 5x5 small cells
    :param thin_cell_line_width: thin line width, for small cells
    :param curve_line_width: line width of the ECG curve
    :param lead_label_font_size: font size of lead labels (aVR, I, II, ...) on ECG image
    :param metadata_font_size: font size of metadata (paper speed, ...) on ECG image
    :param add_metadata: whether metadata (paper speed, ...) should be added at the bottom of the image
    :return: generated ECG image
    """

    # number of timesteps for each lead determined dynamically
    # layout_array should have 3 dimensions: rows, columns, and a third dimension with
    # [lead_idx, start_timestep, stop_timestep]
    # label_array should have 2 dimensions: rows, columns; each entry corresponds to one with the name of the ECG
    # recording_array should have 2 dimensions: timesteps, leads

    rows, cols, _ = layout_array.shape
    calib_rect_small_cell_count = 9
    max_timesteps_per_row = np.max(np.sum(layout_array[:, :, 2] - layout_array[:, :, 1], axis=-1))
    seconds_per_small_cell = 1.0 / paper_speed
    timesteps_per_small_cell = timesteps_per_second * seconds_per_small_cell
    curve_y_scale = (1.0 / millivolts_per_small_cell) * small_cell_size
    calib_rect_width = calib_rect_small_cell_count * small_cell_size

    font = ImageFont.truetype('fonts/courier_prime_bold.ttf', size=lead_label_font_size)
    metadata_font = ImageFont.truetype('fonts/courier_prime_bold.ttf', size=metadata_font_size)

    num_hor_small_cells = math.ceil(max_timesteps_per_row / timesteps_per_small_cell) + cols * calib_rect_small_cell_count
    num_ver_small_cells = rows * vertical_small_cells_per_lead
    img_width = num_hor_small_cells * small_cell_size + thick_cell_line_width
    img_height = num_ver_small_cells * small_cell_size + thick_cell_line_width
    img = Image.new("RGB", (img_width, img_height))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (img_width, img_height)], fill="#ffffff")

    # draw grid

    for run in range(2):
        for i in range(num_hor_small_cells+1):
            is_thick = i % 5 == 0
            if run != int(is_thick):
                continue
            draw.line([(small_cell_size*i, 0), (small_cell_size*i, img_height)],
                      width=thick_cell_line_width if is_thick else thin_cell_line_width,
                      fill=THICK_LINE_COLOR if is_thick else THIN_LINE_COLOR)
        for i in range(num_ver_small_cells+1):
            is_thick = i % 5 == 0
            if run != int(is_thick):
                continue
            draw.line([(0, small_cell_size*i), (img_width, small_cell_size*i)],
                      width=thick_cell_line_width if is_thick else thin_cell_line_width,
                      fill=THICK_LINE_COLOR if is_thick else THIN_LINE_COLOR)

    # draw leads

    acc_idx = 0
    for row_idx in range(rows):
        acc_column_width = 0
        for column_idx in range(cols):
            lead_idx = layout_array[row_idx, column_idx, 0]
            lead_start_timestep = layout_array[row_idx, column_idx, 1]
            lead_end_timestep = layout_array[row_idx, column_idx, 2]
            lead_num_timesteps = lead_end_timestep - lead_start_timestep
            lead_label = label_array[row_idx][column_idx]
            lead_num_small_cells = math.ceil(lead_num_timesteps / timesteps_per_small_cell)
            cell_start_x = acc_column_width
            cell_start_y = row_idx * vertical_small_cells_per_lead * small_cell_size
            cell_width = lead_num_small_cells * small_cell_size + calib_rect_width
            cell_height = vertical_small_cells_per_lead * small_cell_size

            # # mark area belonging to this lead
            # draw.rectangle([(cell_start_x, cell_start_y), (cell_start_x + cell_width, cell_start_y + cell_height)],
            # fill=color_cycle[acc_idx % len(color_cycle)])

            draw.text((small_cell_size // 2 + cell_start_x, small_cell_size // 2 + cell_start_y), lead_label, fill='#000000', align='left', font=font)

            lead_recording = recording_array[lead_start_timestep:lead_end_timestep, lead_idx]

            # draw initial calibration rectangle

            x1 = cell_start_x
            y1 = cell_start_y + (vertical_small_cells_per_lead // 2 + 1) * small_cell_size  # + 1: add some offset for the lead label
            x_square = (x1, x1 + 2 * small_cell_size, x1 + 2 * small_cell_size, x1 + 7 * small_cell_size, x1 + 7 * small_cell_size, x1 + 9 * small_cell_size)
            y_square = (y1, y1, y1 - 10 * small_cell_size, y1 - 10 * small_cell_size, y1, y1)
            draw.line(list(zip(x_square, y_square)), width=curve_line_width, fill="#000000")
            x_shifted = x1 + calib_rect_width

            # draw ECG curve

            inter_point_pixels = (1.0 / timesteps_per_second) * paper_speed * small_cell_size  # [s/timestep * mm/s * pixel/mm = pixel/timestep]
            for idx in range(lead_num_timesteps - 1):
                point_x = x_shifted + idx * inter_point_pixels
                next_point_x = x_shifted + (idx + 1) * inter_point_pixels
                draw.line([(point_x, y1 - curve_y_scale * lead_recording[idx]), (next_point_x, y1 - curve_y_scale * lead_recording[idx + 1])], width=curve_line_width, fill="#000000")

            acc_idx += 1
            acc_column_width += cell_width

    # draw metadata text

    if add_metadata:
        metadata_text = f'{float(paper_speed)} mm/s;  {1.0/millivolts_per_small_cell} mm/mV'
        metadata_start_y = int((rows - 0.25) * vertical_small_cells_per_lead) * small_cell_size
        draw.text((2 * small_cell_size, small_cell_size // 2 + metadata_start_y), metadata_text, fill='#000000',
                  align='left', font=metadata_font)

    return img
