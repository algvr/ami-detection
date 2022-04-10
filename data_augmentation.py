# Sunho Kim, Alexey Gavryushin
# ETH Zurich, 2022


import datasets.ptb_xl.data_handling as ptb_xl_dh
import ecg_plotting
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import random


def generate_example_ecg_image():
    """
    Generate and return an ECG image from a random recording of the PTB-XL dataset
    :return: (image, metadata): a PIL image, and a Pandas series with the single column of metadata associated
    with the ECG recording the image was generated from; the metadata has the same columns as in "ptbxl_database.csv"
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
    img = ecg_plotting.get_ecg_image(layout_array=example_layout_array, label_array=example_label_array,
                                     recording_array=recording)
    return img.convert('RGBA'), Y.iloc[recording_idx]


# based on https://stackoverflow.com/a/68345146
def scale_rotate_translate(image, angle, sr_center=None, displacement=None, scale=None):
    """
    Internal function used to scale, rotate and translate a given image
    :param image: the image to scale, rotate and translate
    :param angle: angle to rotate the image by (in degrees); positive angles rotate the image counterclockwise
    :param sr_center: (unused parameter)
    :param displacement: (unused parameter)
    :param scale: 2-tuple of (vertical, horizontal) factors to scale the image by
    :return: the scaled, rotated and translated image
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

    # expand the image to prevent unintended cropping due to rotation
    new_image = Image.new(image.mode, (large_width, large_height), (0, 0, 0, 0))
    new_image.paste(image, ((large_width - image_width) // 2, (large_height - image_height) // 2), image)

    rotated = new_image.transform((large_width, large_height), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)
    box = rotated.getbbox()  # crop image to content
    cropped = rotated.crop(box)
    return cropped


def get_background_image(idx):
    """
    Return the image with the given index from the "backgrounds" dataset
    :param idx: index of the background image
    :return: image with the given index from the "backgrounds" dataset
    """
    return Image.open('datasets/backgrounds/bg_%.2i' % idx + '.jpg').convert('RGBA')


def get_random_ecg_photo(original_image_scaling_factor_mu=1.0,
                         original_image_scaling_factor_sd=0.0,
                         blur_factor_mu=0.1,
                         blur_factor_sd=0.05,
                         ecg_paper_scale_mu=1.02,
                         ecg_paper_scale_sd=0.02,
                         ecg_paper_y_skew_mu=1.0,
                         ecg_paper_y_skew_sd=0.08,
                         rotation_angle_mu=0.0,
                         rotation_angle_sd=5.0,
                         ecg_paper_relative_translation_x_mu=0.0,
                         ecg_paper_relative_translation_x_sd=0.05,
                         ecg_paper_relative_translation_y_mu=0.0,
                         ecg_paper_relative_translation_y_sd=0.05,
                         shadow_color_beta_1=0.3,
                         shadow_color_beta_2=0.3,
                         shadow_alpha_beta_1=8.0,
                         shadow_alpha_beta_2=25.0,
                         shadow_relative_start_point_mu=0.0,
                         shadow_relative_start_point_sd=0.1,
                         shadow_relative_end_point_mu=1.0,
                         shadow_relative_end_point_sd=0.1,
                         shadow_blur_factor_mu=50.0,
                         shadow_blur_factor_sigma=3.0,
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
    :param shadow_blur_factor_sigma: sigma parameter of Gaussian RV determining the factor by which to blur the shadow
                                     added to the ECG paper in the resulting image
    :param ecg_image: image of the ECG paper to use, or None to generate an ECG image from a random record in the PTB-XL
                      dataset
    :param background_image: background image to use, or None to select a random background image from the "backgrounds"
                             dataset
    :return: image of simulated photograph of an ECG recording
    """
    ecg_data = None
    if ecg_image is None:
        ecg_image, ecg_data = generate_example_ecg_image()
    original_image_scaling_factor =\
        np.clip(np.random.normal(original_image_scaling_factor_mu, original_image_scaling_factor_sd), 0.0, np.infty)
    blur_factor = np.random.normal(blur_factor_mu, blur_factor_sd)
    shadow_relative_start_point = np.random.normal(shadow_relative_start_point_mu, shadow_relative_start_point_sd)
    shadow_relative_end_point = np.random.normal(shadow_relative_end_point_mu, shadow_relative_end_point_sd)
    shadow_blur_factor = np.clip(np.random.normal(shadow_blur_factor_mu, shadow_blur_factor_sigma), 0.0, np.infty)
    ecg_paper_scale = np.random.normal(ecg_paper_scale_mu, ecg_paper_scale_sd)
    ecg_paper_y_skew = np.random.normal(ecg_paper_y_skew_mu, ecg_paper_y_skew_sd)
    rotation_angle = np.random.normal(rotation_angle_mu, rotation_angle_sd)
    ecg_paper_relative_translation_x = np.random.normal(ecg_paper_relative_translation_x_mu,
                                                        ecg_paper_relative_translation_x_sd)
    ecg_paper_relative_translation_y = np.random.normal(ecg_paper_relative_translation_y_mu,
                                                        ecg_paper_relative_translation_y_sd)
    shadow_color = np.random.beta(shadow_color_beta_1, shadow_color_beta_2)
    shadow_alpha = np.random.beta(shadow_alpha_beta_1, shadow_alpha_beta_2)
    ecg_photo = get_ecg_photo_from_image(ecg_image,
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
                                         background_image)
    return ecg_photo, ecg_data


def get_ecg_photo_from_image(ecg_image,
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
                             background_image=None):
    """
    Generate and return a simulated photograph of an ECG recording, using the given arguments as parameters and the
    given ECG image
    :param ecg_image: ECG image to use when generating the photograph
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
    :param background_image: background image to use, or None to select a random background image from the "backgrounds"
                             dataset
    :return: image of simulated photograph of the given ECG recording paper image
    """
    if background_image is None:
        background_image = get_background_image(random.randint(0, 27))

    fg = ecg_image.copy()
    bg = background_image.copy()

    # resize the bg, fg images

    width_bg, height_bg = bg.size
    width_fg, height_fg = fg.size

    size = original_image_scaling_factor * width_bg, original_image_scaling_factor * height_bg
    fg.thumbnail(size, Image.ANTIALIAS)
    width_fg, height_fg = fg.size

    limit_size = 1.5 * width_fg, 1.5*height_fg
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
    rot = scale_rotate_translate(fg, angle=rotation_angle, sr_center=None, displacement=None,
                                 scale=(shearing_factor_vertical, shearing_factor_horizontal))
    width_fg, height_fg = rot.size
    paste_x_coord = int((width_bg - width_fg) / 2 + ecg_paper_relative_translation_x * width_bg)
    paste_y_coord = int((height_bg - height_fg) / 2 + ecg_paper_relative_translation_y * height_bg)

    bg.paste(rot, (paste_x_coord, paste_y_coord), rot)

    # blurring the image
    bg = bg.filter(ImageFilter.GaussianBlur(blur_factor))

    # modify the contrast of the image
    enhancer = ImageEnhance.Contrast(bg)
    bg = enhancer.enhance(1.25)
    return bg
