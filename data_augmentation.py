# Sunho Kim, Alexey Gavryushin
# ETH Zurich, 2022


import datasets.ptb_xl.data_handling as ptb_xl_dh
import ecg_plotting
import math
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import random


def generate_example_ecg_image():
    """
    Generate and return an ECG image from a random recording of the PTB-XL dataset
    :return: (image, mask_images, metadata, lead_pos):
             image: a PIL image containing the result
             mask_images: a list containing mask images of segmentation GT
                          (with classes "ECG curve" at index 0; "thick horizontal line" at 1;
                           "thick vertical line" at 2)
             metadata: a Pandas series with the single column of metadata associated with the ECG recording the image
                       was generated from; the metadata has the same columns as in "ptbxl_database.csv"
             lead_pos: a dict with the positions of the leads (top-left, top-right, bottom-left, bottom-right)
    """
    X, Y = ptb_xl_dh.get_ecg_array(max_samples=100)
    num_recordings, _, _ = X.shape
    recording_idx = random.randint(0, num_recordings-1)
    recording = X[recording_idx]
    labels_lower = list(map(str.lower, ptb_xl_dh.PTB_XL_LEAD_LABELS))
    example_layout_array =\
        np.stack([[[labels_lower.index(ecg_plotting.DEFAULT_ECG_LEAD_SEQUENCE[4 * row_idx + col_idx].lower()), 0, 180]
                   for col_idx in range(4)] for row_idx in range(3)])
    example_label_array = ptb_xl_dh.layout_array_to_label_array(example_layout_array)
    img, mask_images, lead_pos = ecg_plotting.get_ecg_image(layout_array=example_layout_array,
                                                            label_array=example_label_array,
                                                            recording_array=recording)
    return img.convert('RGBA'), mask_images, Y.iloc[recording_idx], lead_pos  # mask_images already RGBA


# based on https://stackoverflow.com/a/68345146
def scale_rotate_translate(image, angle, sr_center=None, displacement=None, scale=None, image_lead_pos=None,
                           crop_bbox=None):
    """
    Internal function used to scale, rotate and translate a given image
    :param image: the image to scale, rotate and translate
    :param angle: angle to rotate the image by (in degrees); positive angles rotate the image counterclockwise
    :param sr_center: (unused parameter)
    :param displacement: (unused parameter)
    :param scale: 2-tuple of (vertical, horizontal) factors to scale the image by
    :param image_lead_pos: positions of leads on the original image (top-left, top-right, bottom-left, bottom-right)
    :param crop_bbox: predefined bounding box to use to crop the intermediate padded image to its contents
                      (pass None to calculate new bounding box)
    :return: tuple with (1) the scaled, rotated and translated image,
                        (2) the lead position list (top-left, top-right, bottom-left, bottom-right)
                        (3) the bounding box used to crop the intermediate padded image to its contents
    """
    # this is needed to avoid cropping due to the rotation of the image
    image_width, image_height = image.size
    large_width = int(1.5 * max(image_width, image_height))
    large_height = int(1.5 * max(image_width, image_height))

    if sr_center is None:
        sr_center = 0, 0
    if displacement is None:
        displacement = 0, 0
    if scale is None:
        scale = 1, 1

    angle = -angle / 180.0 * np.pi

    C = np.array([[1, 0, -sr_center[0]],
                  [0, 1, -sr_center[1]],
                  [0, 0, 1]])

    C_1 = np.linalg.inv(C)

    S = np.array([[scale[0], 0, 0],
                  [0, scale[1], 0],
                  [0,        0, 1]])

    R = np.array([[np.cos(angle), np.sin(angle), 0],
                  [-np.sin(angle), np.cos(angle), 0],
                  [0,                         0, 1]])

    D = np.array([[1, 0, displacement[0]],
                  [0, 1, displacement[1]],
                  [0, 0,            1]])

    Mt = np.dot(D, np.dot(C_1, np.dot(R, np.dot(S, C))))

    a, b, c = Mt[0]
    d, e, f = Mt[1]

    # see https://stackoverflow.com/a/17141975:
    # Data [(3rd input arg to new_image.transform)] is a 6-tuple (a, b, c, d, e, f) which contains the first two rows
    # from an affine transform matrix. For each pixel (x, y) in the output image, the new value is taken from a position
    # (a x + b y + c, d x + e y + f) in the input image, rounded to nearest pixel.

    # xold = a xnew + b ynew + c
    # yold = d xnew + e ynew + f
    # --> xnew = (xold - b ynew - c) / a
    # --> ynew = (yold - d xnew - f) / e

    # --> xnew = (xold - b ((yold - d xnew - f) / e) - c) / a
    #          = 1/a xold - b/(ea) yold + bd/(ea) xnew + bf/ea - c/a
    # (1 - bd/(ea)) xnew = 1/a xold - b/(ea) yold + bf/ea - c/a
    # xnew = (1/a xold - b/(ea) yold + bf/ea - c/a) / (1 - bd/(ea))

    # ynew = (yold - d xnew - f) / e

    transform_x = lambda x, y: ((1.0/a) * x - b/(e*a) * y + (b*f)/(e*a) - c/a) / (1.0 - (b*d)/(e*a))
    transform_y = lambda x, y: (y - d * transform_x(x, y) - f) / e
    paste_x = (large_width - image_width) // 2
    paste_y = (large_height - image_height) // 2

    if image_lead_pos is not None:
        new_lead_pos = {}
        for key, pos in image_lead_pos.items():
            new_lead_pos[key] = (transform_x(paste_x + pos[0], paste_y + pos[1]),
                                 transform_y(paste_x + pos[0], paste_y + pos[1]),
                                 transform_x(paste_x + pos[2], paste_y + pos[3]),
                                 transform_y(paste_x + pos[2], paste_y + pos[3]),
                                 transform_x(paste_x + pos[4], paste_y + pos[5]),
                                 transform_y(paste_x + pos[4], paste_y + pos[5]),
                                 transform_x(paste_x + pos[6], paste_y + pos[7]),
                                 transform_y(paste_x + pos[6], paste_y + pos[7]))
    else:
        new_lead_pos = None

    # expand the image to prevent unintended cropping due to rotation
    new_image = Image.new(image.mode, (large_width, large_height), (0, 0, 0, 0))
    new_image.paste(image, (paste_x, paste_y), image)

    rotated = new_image.transform((large_width, large_height), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)
    if crop_bbox is not None:
        box = crop_bbox
    else:
        box = rotated.getbbox()  # crop image to content
    cropped = rotated.crop(box)

    if new_lead_pos is not None:
        delta_x = box[0]
        delta_y = box[1]
        for key, pos in new_lead_pos.items():
            new_lead_pos[key] = (pos[0] - delta_x,
                                 pos[1] - delta_y,
                                 pos[2] - delta_x,
                                 pos[3] - delta_y,
                                 pos[4] - delta_x,
                                 pos[5] - delta_y,
                                 pos[6] - delta_x,
                                 pos[7] - delta_y)

    return cropped, new_lead_pos, box


def get_background_image(idx):
    """
    Return the image with the given index from the "backgrounds" dataset
    :param idx: index of the background image
    :return: image with the given index from the "backgrounds" dataset
    """
    return Image.open('datasets/backgrounds/bg_%.2i' % idx + '.jpg').convert('RGBA')


ECG_GEN_DEFAULT_ORIGINAL_IMAGE_SCALING_FACTOR_MU = 1.0
ECG_GEN_DEFAULT_ORIGINAL_IMAGE_SCALING_FACTOR_SD = 0.0
ECG_GEN_DEFAULT_BLUR_FACTOR_MU = 0.1
ECG_GEN_DEFAULT_BLUR_FACTOR_SD = 0.05
ECG_GEN_DEFAULT_PAPER_SCALE_MU = 1.02
ECG_GEN_DEFAULT_PAPER_SCALE_SD = 0.02
ECG_GEN_DEFAULT_PAPER_Y_SKEW_MU = 1.0
ECG_GEN_DEFAULT_PAPER_Y_SKEW_SD = 0.08
ECG_GEN_DEFAULT_ROTATION_ANGLE_MU = 0.0
ECG_GEN_DEFAULT_ROTATION_ANGLE_SD = 5.0
ECG_GEN_DEFAULT_PAPER_RELATIVE_TRANSLATION_X_MU = 0.0
ECG_GEN_DEFAULT_PAPER_RELATIVE_TRANSLATION_X_SD = 0.05
ECG_GEN_DEFAULT_PAPER_RELATIVE_TRANSLATION_Y_MU = 0.0
ECG_GEN_DEFAULT_PAPER_RELATIVE_TRANSLATION_Y_SD = 0.05
ECG_GEN_DEFAULT_SHADOW_COLOR_BETA_1 = 0.3
ECG_GEN_DEFAULT_SHADOW_COLOR_BETA_2 = 0.3
ECG_GEN_DEFAULT_SHADOW_ALPHA_BETA_1 = 8.0
ECG_GEN_DEFAULT_SHADOW_ALPHA_BETA_2 = 25.0
ECG_GEN_DEFAULT_SHADOW_RELATIVE_START_POINT_MU = 0.0
ECG_GEN_DEFAULT_SHADOW_RELATIVE_START_POINT_SD = 0.1
ECG_GEN_DEFAULT_SHADOW_RELATIVE_END_POINT_MU = 1.0
ECG_GEN_DEFAULT_SHADOW_RELATIVE_END_POINT_SD = 0.1
ECG_GEN_DEFAULT_SHADOW_BLUR_FACTOR_MU = 50.0
ECG_GEN_DEFAULT_SHADOW_BLUR_FACTOR_SD = 3.0
ECG_GEN_DEFAULT_WHITE_NOISE_P_BETA_1 = 5.0
ECG_GEN_DEFAULT_WHITE_NOISE_P_BETA_2 = 5.0
ECG_GEN_DEFAULT_WHITE_NOISE_SD_MU = 20.0
ECG_GEN_DEFAULT_WHITE_NOISE_SD_SD = 5.0


def get_default_ecg_param_dict():
    """
    Return a dictionary of parameters and their default values for the "get_random_ecg_photo" function.
    """
    return {'original_image_scaling_factor_mu': ECG_GEN_DEFAULT_ORIGINAL_IMAGE_SCALING_FACTOR_MU,
            'original_image_scaling_factor_sd': ECG_GEN_DEFAULT_ORIGINAL_IMAGE_SCALING_FACTOR_SD,
            'blur_factor_mu': ECG_GEN_DEFAULT_BLUR_FACTOR_MU,
            'blur_factor_sd': ECG_GEN_DEFAULT_BLUR_FACTOR_SD,
            'ecg_paper_scale_mu': ECG_GEN_DEFAULT_PAPER_SCALE_MU,
            'ecg_paper_scale_sd': ECG_GEN_DEFAULT_PAPER_SCALE_SD,
            'ecg_paper_y_skew_mu': ECG_GEN_DEFAULT_PAPER_Y_SKEW_MU,
            'ecg_paper_y_skew_sd': ECG_GEN_DEFAULT_PAPER_Y_SKEW_SD,
            'rotation_angle_mu': ECG_GEN_DEFAULT_ROTATION_ANGLE_MU,
            'rotation_angle_sd': ECG_GEN_DEFAULT_ROTATION_ANGLE_SD,
            'ecg_paper_relative_translation_x_mu': ECG_GEN_DEFAULT_PAPER_RELATIVE_TRANSLATION_X_MU,
            'ecg_paper_relative_translation_x_sd': ECG_GEN_DEFAULT_PAPER_RELATIVE_TRANSLATION_X_SD,
            'ecg_paper_relative_translation_y_mu': ECG_GEN_DEFAULT_PAPER_RELATIVE_TRANSLATION_Y_MU,
            'ecg_paper_relative_translation_y_sd': ECG_GEN_DEFAULT_PAPER_RELATIVE_TRANSLATION_Y_SD,
            'shadow_color_beta_1': ECG_GEN_DEFAULT_SHADOW_COLOR_BETA_1,
            'shadow_color_beta_2': ECG_GEN_DEFAULT_SHADOW_COLOR_BETA_2,
            'shadow_alpha_beta_1': ECG_GEN_DEFAULT_SHADOW_ALPHA_BETA_1,
            'shadow_alpha_beta_2': ECG_GEN_DEFAULT_SHADOW_ALPHA_BETA_2,
            'shadow_relative_start_point_mu': ECG_GEN_DEFAULT_SHADOW_RELATIVE_START_POINT_MU,
            'shadow_relative_start_point_sd': ECG_GEN_DEFAULT_SHADOW_RELATIVE_START_POINT_SD,
            'shadow_relative_end_point_mu': ECG_GEN_DEFAULT_SHADOW_RELATIVE_END_POINT_MU,
            'shadow_relative_end_point_sd': ECG_GEN_DEFAULT_SHADOW_RELATIVE_END_POINT_SD,
            'shadow_blur_factor_mu': ECG_GEN_DEFAULT_SHADOW_BLUR_FACTOR_MU,
            'shadow_blur_factor_sd': ECG_GEN_DEFAULT_SHADOW_BLUR_FACTOR_SD,
            'white_noise_p_beta_1': ECG_GEN_DEFAULT_WHITE_NOISE_P_BETA_1,
            'white_noise_p_beta_2': ECG_GEN_DEFAULT_WHITE_NOISE_P_BETA_2,
            'white_noise_sd_mu': ECG_GEN_DEFAULT_WHITE_NOISE_SD_MU,
            'white_noise_sd_sd': ECG_GEN_DEFAULT_WHITE_NOISE_SD_SD}


def get_random_ecg_photo(original_image_scaling_factor_mu=ECG_GEN_DEFAULT_ORIGINAL_IMAGE_SCALING_FACTOR_MU,
                         original_image_scaling_factor_sd=ECG_GEN_DEFAULT_ORIGINAL_IMAGE_SCALING_FACTOR_SD,
                         blur_factor_mu=ECG_GEN_DEFAULT_BLUR_FACTOR_MU,
                         blur_factor_sd=ECG_GEN_DEFAULT_BLUR_FACTOR_SD,
                         ecg_paper_scale_mu=ECG_GEN_DEFAULT_PAPER_SCALE_MU,
                         ecg_paper_scale_sd=ECG_GEN_DEFAULT_PAPER_SCALE_SD,
                         ecg_paper_y_skew_mu=ECG_GEN_DEFAULT_PAPER_Y_SKEW_MU,
                         ecg_paper_y_skew_sd=ECG_GEN_DEFAULT_PAPER_Y_SKEW_SD,
                         rotation_angle_mu=ECG_GEN_DEFAULT_ROTATION_ANGLE_MU,
                         rotation_angle_sd=ECG_GEN_DEFAULT_ROTATION_ANGLE_SD,
                         ecg_paper_relative_translation_x_mu=ECG_GEN_DEFAULT_PAPER_RELATIVE_TRANSLATION_X_MU,
                         ecg_paper_relative_translation_x_sd=ECG_GEN_DEFAULT_PAPER_RELATIVE_TRANSLATION_X_SD,
                         ecg_paper_relative_translation_y_mu=ECG_GEN_DEFAULT_PAPER_RELATIVE_TRANSLATION_Y_MU,
                         ecg_paper_relative_translation_y_sd=ECG_GEN_DEFAULT_PAPER_RELATIVE_TRANSLATION_Y_SD,
                         shadow_color_beta_1=ECG_GEN_DEFAULT_SHADOW_COLOR_BETA_1,
                         shadow_color_beta_2=ECG_GEN_DEFAULT_SHADOW_COLOR_BETA_2,
                         shadow_alpha_beta_1=ECG_GEN_DEFAULT_SHADOW_ALPHA_BETA_1,
                         shadow_alpha_beta_2=ECG_GEN_DEFAULT_SHADOW_ALPHA_BETA_2,
                         shadow_relative_start_point_mu=ECG_GEN_DEFAULT_SHADOW_RELATIVE_START_POINT_MU,
                         shadow_relative_start_point_sd=ECG_GEN_DEFAULT_SHADOW_RELATIVE_START_POINT_SD,
                         shadow_relative_end_point_mu=ECG_GEN_DEFAULT_SHADOW_RELATIVE_END_POINT_MU,
                         shadow_relative_end_point_sd=ECG_GEN_DEFAULT_SHADOW_RELATIVE_END_POINT_SD,
                         shadow_blur_factor_mu=ECG_GEN_DEFAULT_SHADOW_BLUR_FACTOR_MU,
                         shadow_blur_factor_sd=ECG_GEN_DEFAULT_SHADOW_BLUR_FACTOR_SD,
                         white_noise_p_beta_1=ECG_GEN_DEFAULT_WHITE_NOISE_P_BETA_1,
                         white_noise_p_beta_2=ECG_GEN_DEFAULT_WHITE_NOISE_P_BETA_2,
                         white_noise_sd_mu=ECG_GEN_DEFAULT_WHITE_NOISE_SD_MU,
                         white_noise_sd_sd=ECG_GEN_DEFAULT_WHITE_NOISE_SD_SD,
                         ecg_image=None,
                         background_image=None):
    """
    Generate and return a simulated photograph of an ECG recording, using randomly generated parameters with
    distributions parameterized by the given arguments
    :param original_image_scaling_factor_mu: mu parameter of Gaussian RV determining the factor by which to scale the
                                             resulting image
    :param original_image_scaling_factor_sd: sigma parameter of Gaussian RV determining the factor by which to scale the
                                             resulting image
    :param blur_factor_mu: mu parameter of Gaussian RV determining the factor by which to blur the ECG paper in the
                           resulting image
    :param blur_factor_sd: sigma parameter of Gaussian RV determining the factor by which to blur the ECG paper in the
                           resulting image
    :param ecg_paper_scale_mu: mu parameter of Gaussian RV determining the factor by which to scale the ECG paper in
                               the resulting image
    :param ecg_paper_scale_sd: sigma parameter of Gaussian RV determining the factor by which to scale the ECG paper in
                               the resulting image
    :param ecg_paper_y_skew_mu: mu parameter of Gaussian RV determining the factor by which to vertically skew the ECG
                                paper in the resulting image in relation to the horizontal axis
    :param ecg_paper_y_skew_sd: sigma parameter of Gaussian RV determining the factor by which to vertically skew the
                                ECG paper in the resulting image in relation to the horizontal axis
    :param rotation_angle_mu: mu parameter of Gaussian RV determining the factor by which to rotate counterclockwise
                              the ECG paper in the resulting image
    :param rotation_angle_sd: sigma parameter of Gaussian RV determining the factor by which to rotate counterclockwise
                              the ECG paper in the resulting image
    :param ecg_paper_relative_translation_x_mu: mu parameter of Gaussian RV determining the proportion of the total
                                                image width by which to horizontally translate the ECG paper in the
                                                resulting image
    :param ecg_paper_relative_translation_x_sd: sigma parameter of Gaussian RV determining the proportion of the total
                                                image width by which to horizontally translate the ECG paper in the
                                                resulting image
    :param ecg_paper_relative_translation_y_mu: mu parameter of Gaussian RV determining the proportion of the total
                                                image height by which to vertically translate the ECG paper in the
                                                resulting image
    :param ecg_paper_relative_translation_y_sd: sigma parameter of Gaussian RV determining the proportion of the total
                                                image height by which to vertically translate the ECG paper in the
                                                resulting image
    :param shadow_color_beta_1: alpha parameter of Beta-distributed RV determining the brightness of the shadow to add
                                to the ECG paper in the resulting image

    :param shadow_color_beta_2: beta parameter of Beta-distributed (0.0 - 1.0) RV determining the brightness of
                                the shadow to add to the ECG paper in the resulting image
    :param shadow_alpha_beta_1: alpha parameter of Beta-distributed (0.0 - 1.0) RV determining the opacity (alpha) of
                                the shadow to add to the ECG paper in the resulting image
    :param shadow_alpha_beta_2: alpha parameter of Beta-distributed RV determining the opacity (alpha) of the shadow to
                                add to the ECG paper in the resulting image
    :param shadow_relative_start_point_mu: mu parameter of Gaussian RV determining the starting position P_s of the
                                           shadow on the ECG paper in the image relative to the width and height of the
                                           image, with the shadow starting at
                                           (x = P_s*image_width, y = P_s*image_height)
    :param shadow_relative_start_point_sd: sigma parameter of Gaussian RV determining the starting position P_s of the
                                           shadow on the ECG paper in the image relative to the width and height of the
                                           image, with the shadow starting at
                                           (x = P_s*image_width, y = P_s*image_height)
    :param shadow_relative_end_point_mu: mu parameter of Gaussian RV determining the ending position P_e of the
                                         shadow on the ECG paper in the image relative to the width and height of the
                                         image, with the shadow ending at
                                         (x = P_e*image_width, y = P_e*image_height)
    :param shadow_relative_end_point_sd: sigma parameter of Gaussian RV determining the ending position P_e of the
                                         shadow on the ECG paper in the image relative to the width and height of the
                                         image, with the shadow ending at
                                         (x = P_e*image_width, y = P_e*image_height)
    :param shadow_blur_factor_mu: mu parameter of Gaussian RV determining the factor by which to blur the shadow
                                  added to the ECG paper in the resulting image
    :param shadow_blur_factor_sd: sigma parameter of Gaussian RV determining the factor by which to blur the shadow
                                     added to the ECG paper in the resulting image
    :param white_noise_p_beta_1:  alpha parameter of Beta distribution (0.0 - 1.0) from which to draw the P parameter
                                  of the Bernoulli distribution determining whether white noise is to be added to a
                                  given pixel
    :param white_noise_p_beta_2:  beta parameter of Beta distribution (0.0 - 1.0) from which to draw the P parameter
                                  of the Bernoulli distribution determining whether white noise is to be added to a
                                  given pixel
    :param white_noise_sd_mu: mu parameter of Gaussian distribution from which to draw the sigma parameter of the
                                 Gaussian RV determining the intensity of white noise to add to a given pixel
    :param white_noise_sd_sd: sigma parameter of Gaussian distribution from which to draw the sigma parameter of
                              the Gaussian RV determining the intensity of white noise to add to a given pixel
    :param ecg_image: image of the ECG paper to use, or None to generate an ECG image from a random record in the PTB-XL
                      dataset
    :param background_image: background image to use, or None to select a random background image from the "backgrounds"
                             dataset
    :return: image of simulated photograph of an ECG recording, transformed mask images, metadata of associated ECG,
             dict with positions of leads on image (top-left, top-right, bottom-left, bottom-right), and angle by which
             the ECG image was rotated
    """
    ecg_data = None
    if ecg_image is None:
        ecg_image, mask_images, ecg_data, image_lead_pos = generate_example_ecg_image()
    original_image_scaling_factor =\
        np.clip(np.random.normal(original_image_scaling_factor_mu, original_image_scaling_factor_sd), 0.0, np.infty)
    blur_factor = np.random.normal(blur_factor_mu, blur_factor_sd)
    shadow_relative_start_point = np.random.normal(shadow_relative_start_point_mu, shadow_relative_start_point_sd)
    shadow_relative_end_point = np.random.normal(shadow_relative_end_point_mu, shadow_relative_end_point_sd)
    shadow_blur_factor = np.clip(np.random.normal(shadow_blur_factor_mu, shadow_blur_factor_sd), 0.0, np.infty)
    ecg_paper_scale = np.random.normal(ecg_paper_scale_mu, ecg_paper_scale_sd)
    ecg_paper_y_skew = np.random.normal(ecg_paper_y_skew_mu, ecg_paper_y_skew_sd)
    rotation_angle = np.random.normal(rotation_angle_mu, rotation_angle_sd)
    ecg_paper_relative_translation_x = np.random.normal(ecg_paper_relative_translation_x_mu,
                                                        ecg_paper_relative_translation_x_sd)
    ecg_paper_relative_translation_y = np.random.normal(ecg_paper_relative_translation_y_mu,
                                                        ecg_paper_relative_translation_y_sd)
    shadow_color = np.random.beta(shadow_color_beta_1, shadow_color_beta_2)
    shadow_alpha = np.random.beta(shadow_alpha_beta_1, shadow_alpha_beta_2)
    white_noise_p = np.random.beta(white_noise_p_beta_1, white_noise_p_beta_2)
    white_noise_sd = np.random.normal(white_noise_sd_mu, white_noise_sd_sd)
    ecg_photo, new_mask_images, photo_lead_pos = get_ecg_photo_from_image(ecg_image,
                                                                          mask_images,
                                                                          original_image_scaling_factor,
                                                                          blur_factor,
                                                                          ecg_paper_scale,
                                                                          ecg_paper_y_skew,
                                                                          rotation_angle,
                                                                          ecg_paper_relative_translation_x,
                                                                          ecg_paper_relative_translation_y,
                                                                          shadow_color,
                                                                          shadow_alpha,
                                                                          shadow_relative_start_point,
                                                                          shadow_relative_end_point,
                                                                          shadow_blur_factor,
                                                                          white_noise_p,
                                                                          white_noise_sd,
                                                                          background_image,
                                                                          image_lead_pos)
    return ecg_photo, new_mask_images, ecg_data, photo_lead_pos, rotation_angle


def get_ecg_photo_from_image(ecg_image,
                             mask_images,
                             original_image_scaling_factor,
                             blur_factor,
                             ecg_paper_scale,
                             ecg_paper_y_skew,
                             rotation_angle,
                             ecg_paper_relative_translation_x,
                             ecg_paper_relative_translation_y,
                             shadow_color,
                             shadow_alpha,
                             shadow_relative_start_point,
                             shadow_relative_end_point,
                             shadow_blur_factor,
                             white_noise_p,
                             white_noise_sd,
                             background_image=None,
                             image_lead_pos=None):
    """
    Generate and return a simulated photograph of an ECG recording, using the given arguments as parameters and the
    given ECG image
    :param ecg_image: ECG image to use when generating the photograph
    :param mask_images: a list containing mask images of segmentation GT
                        (with classes "ECG curve" at index 0; "thick horizontal line" at 1;
                         1"thick vertical line" at 2)
    :param image_lead_pos: positions of leads on image (or None;
                                                        format: (top-left, top-right, bottom-left, bottom-right))
    :param original_image_scaling_factor: factor by which to scale the resulting image
    :param blur_factor: factor by which to blur the ECG paper in the resulting image
    :param ecg_paper_scale: factor by which to scale the ECG paper in the resulting image
    :param ecg_paper_y_skew: factor by which to vertically skew the ECG paper in the resulting image in relation to the
                             horizontal axis
    :param rotation_angle: factor by which to rotate counterclockwise the ECG paper in the resulting image
    :param ecg_paper_relative_translation_x: proportion of the total image width by which to horizontally translate the
                                             ECG paper in the resulting image
    :param ecg_paper_relative_translation_y: proportion of the total image height by which to vertically translate the
                                             ECG paper in the resulting image
    :param shadow_color: brightness of the shadow to add to the ECG paper in the resulting image (0.0 - 1.0)
    :param shadow_alpha: opacity (alpha) of the shadow to add to the ECG paper in the resulting image (0.0 - 1.0)
    :param shadow_relative_start_point: starting position P_s of the shadow on the ECG paper in the image relative to
                                        the width and height of the image, with the shadow starting at
                                        (x = P_s*image_width, y = P_s*image_height)
    :param shadow_relative_end_point: ending position P_e of the shadow on the ECG paper in the image relative to the
                                      width and height of the image, with the shadow ending at
                                      (x = P_e*image_width, y = P_e*image_height)
    :param shadow_blur_factor: factor by which to blur the shadow added to the ECG paper in the resulting image
    :param white_noise_p: P parameter of Bernoulli distribution determining whether white noise is to be added to a
                          given pixel
    :param white_noise_sd: sigma parameter of Gaussian RV determining the intensity of white noise to add to a
                           given pixel
    :param background_image: background image to use, or None to select a random background image from the "backgrounds"
                             dataset
    :return: image of simulated photograph of the given ECG recording paper image, transformed mask images,
             dict with positions of leads on image (top-left, top-right, bottom-left, bottom-right)
    """

    # !!!!!!!!!! TODO: remove! !!!!!!!!!!
    # ecg_image.convert('RGB').save('__img.jpg')
    # [img_obj.save(f'__{idx}.png') for idx, img_obj in enumerate(mask_images)]

    new_mask_image_fgs = [img_obj.copy() for img_obj in mask_images]
    
    if background_image is None:
        background_image = get_background_image(random.randint(0, 27))

    fg = ecg_image.copy()
    bg = background_image.copy()

    # resize the bg, fg images

    width_bg, height_bg = bg.size
    width_fg, height_fg = fg.size
    
    width_fg_orig, height_fg_orig = fg.size

    size = original_image_scaling_factor * width_bg, original_image_scaling_factor * height_bg
    for img_obj in [fg, *new_mask_image_fgs]:
        img_obj.thumbnail(size, Image.ANTIALIAS)
    width_fg, height_fg = fg.size

    # rescale
    if image_lead_pos is not None:
        image_lead_pos_scaled = {}
        lead_scale_x = width_fg / width_fg_orig
        lead_scale_y = height_fg / height_fg_orig
        for key, pos in image_lead_pos.items():
            image_lead_pos_scaled[key] = (pos[0] * lead_scale_x, pos[1] * lead_scale_y,
                                          pos[2] * lead_scale_x, pos[3] * lead_scale_y,
                                          pos[4] * lead_scale_x, pos[5] * lead_scale_y,
                                          pos[6] * lead_scale_x, pos[7] * lead_scale_y)

    else:
        image_lead_pos_scaled = None

    # do not copy the background; instead, initialize a transparent image with the same size as the background

    limit_size = 1.5 * width_fg, 1.5 * height_fg
    bg.thumbnail(limit_size, Image.ANTIALIAS)
    width_bg, height_bg = bg.size

    # shadow on the fg
    transp = Image.new('RGBA', fg.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(transp, 'RGBA')
    shadow_color_int = int(shadow_color * 255)
    shadow_alpha_int = int(shadow_alpha * 255)
    draw.ellipse((shadow_relative_start_point * width_fg, shadow_relative_start_point * height_fg,
                  shadow_relative_end_point * width_fg, shadow_relative_end_point * height_fg),
                 fill=(shadow_color_int, shadow_color_int, shadow_color_int, shadow_alpha_int))
    transp = transp.filter(ImageFilter.GaussianBlur(shadow_blur_factor))
    
    fg.paste(Image.alpha_composite(fg, transp))

    # shearing the image
    shearing_factor_vertical = ecg_paper_scale * ecg_paper_y_skew
    shearing_factor_horizontal = ecg_paper_scale

    crop_bbox = None

    def srt(img):
        nonlocal crop_bbox
        _rot, _photo_lead_pos, crop_bbox =\
            scale_rotate_translate(img, angle=rotation_angle, sr_center=None,
                                   displacement=None, scale=(shearing_factor_vertical, shearing_factor_horizontal),
                                   image_lead_pos=image_lead_pos_scaled, crop_bbox=crop_bbox)
        return _rot, _photo_lead_pos

    rot, photo_lead_pos = srt(fg)

    for img_idx, img_obj in enumerate(new_mask_image_fgs):
        rot_img_obj, _ = srt(img_obj)
        new_mask_image_fgs[img_idx] = rot_img_obj
    
    width_fg, height_fg = rot.size
    paste_x_coord = int((width_bg - width_fg) / 2 + ecg_paper_relative_translation_x * width_bg)
    paste_y_coord = int((height_bg - height_fg) / 2 + ecg_paper_relative_translation_y * height_bg)

    bg.paste(rot, (paste_x_coord, paste_y_coord), rot)

    new_mask_image_bgs = [Image.new('RGBA', bg.size, (0, 0, 0, 0)) for _ in range(len(mask_images))]
    for img_idx, img_obj in enumerate(new_mask_image_fgs):
        new_mask_image_bgs[img_idx].paste(img_obj, (paste_x_coord, paste_y_coord), img_obj)

    # rescale again
    if photo_lead_pos is not None:
        for key, pos in photo_lead_pos.items():
            photo_lead_pos[key] = (round(pos[0] + paste_x_coord), round(pos[1] + paste_y_coord),
                                   round(pos[2] + paste_x_coord), round(pos[3] + paste_y_coord),
                                   round(pos[4] + paste_x_coord), round(pos[5] + paste_y_coord),
                                   round(pos[6] + paste_x_coord), round(pos[7] + paste_y_coord))

    # blurring the image
    bg = bg.filter(ImageFilter.GaussianBlur(blur_factor))

    # modify the contrast of the image
    enhancer = ImageEnhance.Contrast(bg)
    bg = enhancer.enhance(1.25)

    # adding white noise
    if white_noise_p > 0.0:
        bg_rgb = bg.convert('RGB')
        num_channels = len(bg_rgb.getbands())
        img_shape = (height_bg, width_bg, num_channels)
        white_noise_mask = np.random.binomial(n=1, p=white_noise_p, size=img_shape)
        white_noise = np.random.normal(0, white_noise_sd, size=img_shape)
        img_array = np.clip(np.array(bg_rgb) + white_noise_mask * white_noise, 0, 255).astype(np.uint8)
        bg = Image.fromarray(img_array)

    return bg, new_mask_image_bgs, photo_lead_pos
