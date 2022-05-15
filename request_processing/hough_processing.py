import cv2
import numpy as np
import os
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans, MeanShift
from sklearn.linear_model import RANSACRegressor, LinearRegression


NUM_LINE_CLUSTERS = 5

# 0 ~ vertical line, pi/2 ~ horizontal line

HOR_LINE_RANGE_MEDIUM = (np.pi / 2.0 - 45.0 * (np.pi / 180.0), np.pi / 2.0 + 45.0 * (np.pi / 180.0))
HOR_LINE_RANGE_SMALL = (np.pi / 2.0 - 20.0 * (np.pi / 180.0), np.pi / 2.0 + 20.0 * (np.pi / 180.0))

VERT_LINE_RANGE_MEDIUM = (-45.0 * (np.pi / 180.0), +45.0 * (np.pi / 180.0))
VERT_LINE_RANGE_SMALL = (-20.0 * (np.pi / 180.0), +20.0 * (np.pi / 180.0))

def pillow_to_cv2(pil_img):
    cv2_img = np.array(pil_img)
    # Convert RGB to BGR
    if len(cv2_img.shape) == 3:
        cv2_img = cv2_img[:, :, ::-1].copy()
    else:
        cv2_img = cv2_img[:, :].copy()
    return cv2_img


def detect_lines(cv2_img):
    if len(cv2_img.shape) == 3:
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2_img
    edges = cv2.Canny(gray, 10, 100, apertureSize=7)
    lines = cv2.HoughLines(edges, 2, (np.pi / 180), 200)
    return np.squeeze(lines)


def get_matching_cluster(k_means_obj, line_range, num_line_clusters=NUM_LINE_CLUSTERS):
    cluster_votes = np.bincount(k_means_obj.labels_)
    cluster_angles = np.squeeze(k_means_obj.cluster_centers_)
    max_cluster_angle = np.inf
    max_cluster_votes = 0
    max_cluster_idx = -1

    for this_cluster_idx, this_cluster_angle, this_cluster_votes in zip(list(range(num_line_clusters)),
                                                                        cluster_angles,
                                                                        cluster_votes):
        if line_range[0] <= this_cluster_angle <= line_range[1] and max_cluster_votes < this_cluster_votes:
            max_cluster_idx = this_cluster_idx
            max_cluster_votes = this_cluster_votes
            max_cluster_angle = this_cluster_angle
    if max_cluster_idx == -1:
        # found no lines; error out
        raise RuntimeError('No lines detected in supplied image during "get_matching_cluster"!')

    return max_cluster_idx, max_cluster_angle, max_cluster_votes


def get_deg_rot_angle_loop(cv2_img, log_dict=None, log_dict_suffix=None):
    # heavily based on detect_hor_vert_cell_distances_loop
    # we only care about horizontal lines here (thus, MODE_VERT)
    
    log_img = Image.fromarray(np.copy(cv2_img) if len(cv2_img.shape) == 3\
                              else np.repeat(np.expand_dims(cv2_img, axis=-1), 3, axis=-1))
                              
    cv2_img_height = cv2_img.shape[0]
    cv2_img_width = cv2_img.shape[1]

    # here, we should actually probe a lot of slices (columns), to be more robust
    NUM_SLICES_TO_PROBE = 50

    MODE_HOR = 0
    MODE_VERT = 1

    STATE_OUTSIDE_LINE = 0
    STATE_INSIDE_LINE = 1
    for mode in [MODE_VERT]:  # 0: horizontal; 1: vertical
        # when looking for hor dist, detect vert lines, and vice versa
        mode_img_value_to_detect = {MODE_HOR: 85, MODE_VERT: 170}[mode]
        bg_class = 0

        probe_dim_value = cv2_img_height if mode == MODE_HOR else cv2_img_width
        scanthrough_dim_value = cv2_img_width if mode == MODE_HOR else cv2_img_height
        slices_to_scan_idxs = []
        for _slice_to_scan_idx in range(0, probe_dim_value, int(probe_dim_value / NUM_SLICES_TO_PROBE)):
            slice_to_scan_idx = _slice_to_scan_idx
            while slice_to_scan_idx in slices_to_scan_idxs and slice_to_scan_idx < probe_dim_value:
                slice_to_scan_idx += 1
            if slice_to_scan_idx < probe_dim_value:
                slices_to_scan_idxs.append(slice_to_scan_idx)

        slice_line_start_idxs_collector = []
        slice_line_start_idxs_slice_idxs = []

        largest_slice_line_start_idxs_length = 0
        largest_slice_line_start_idxs_slice_idx = -1

        for slice_to_scan_idx in slices_to_scan_idxs:
            state = STATE_OUTSIDE_LINE
            MIN_INTER_LINE_DISTANCE = 5
            MIN_LINE_EXIT_DISTANCE = 5
            # we'll just assume that there are no/very little false positives
            # go through slice; once we find a "line pixel", switch to STATE_INSIDE_LINE; once we leave, switch to 
            # STATE_OUTSIDE_LINE; if another line has been found in the range of the last MIN_INTER_LINE_DISTANCE pixels
            # do not trigger a new line again
            # (be careful here to respect all state switches)

            # note that we might accidentally hit a whole line of the other orientation
            # -> only consider pixels of the class we are looking for
            # -> however, some patches might correspond to intersections of lines, or the ECG curve may intersect
            #    some lines
            # -> if we are *in* a line, we should only leave that line if we encounter a background pixel, not any other
            #    kind of pixel

            slice_line_start_idxs = []
            slice_last_line_element_idx = -np.inf
            for slice_element_idx in range(scanthrough_dim_value):
                pixel_x = slice_element_idx if mode == MODE_HOR else slice_to_scan_idx
                pixel_y = slice_to_scan_idx if mode == MODE_HOR else slice_element_idx
                pixel_value = cv2_img[pixel_y, pixel_x]

                # cv2_img has elements from 0 to 255;
                # if pixel_value == 85, it's a hor line; if pixel_value == 170, it's a vert line
                last_line_start = -np.inf if len(slice_line_start_idxs) == 0 else slice_line_start_idxs[-1]

                # the lines may be very thin, even just one pixel wide
                # hence, we immediately enter STATE_INSIDE_LINE whenever we encounter a white pixel

                # however, we assume that the lines have at least some minimum amount of distance between them

                if state == STATE_OUTSIDE_LINE\
                   and pixel_value == mode_img_value_to_detect\
                   and (slice_element_idx - last_line_start) >= MIN_INTER_LINE_DISTANCE:
                    state = STATE_INSIDE_LINE
                    # log_img.putpixel((pixel_x, pixel_y), (0, 255, 0) if mode == MODE_HOR else (0, 0, 255))
                    slice_line_start_idxs.append(slice_element_idx)
                elif state == STATE_INSIDE_LINE\
                    and pixel_value == bg_class and slice_last_line_element_idx >= MIN_LINE_EXIT_DISTANCE:
                    # for now, we don't use the exit points of the lines for anything
                    # we just allow new lines to be detected by entering STATE_OUTSIDE_LINE again
                    state = STATE_OUTSIDE_LINE

                if state == STATE_INSIDE_LINE:
                    slice_last_line_element_idx = slice_element_idx
                    # log_img.putpixel((pixel_x, pixel_y), (0, 255, 0) if mode == MODE_HOR else (0, 0, 255))

            if len(slice_line_start_idxs) > largest_slice_line_start_idxs_length:
                largest_slice_line_start_idxs_length = len(slice_line_start_idxs)
                largest_slice_line_start_idxs_slice_idx = slice_to_scan_idx
            slice_line_start_idxs_collector.append(slice_line_start_idxs)
            slice_line_start_idxs_slice_idxs.append(slice_to_scan_idx)
            
        # mean_inter_line_distance = np.mean(inter_line_distances_np)
        # median_inter_line_distance = np.median(inter_line_distances_np)

        #if mode == MODE_HOR:
        #    mean_hor_dist = mean_inter_line_distance
        #    median_hor_dist = median_inter_line_distance
        #else:
        #    mean_vert_dist = mean_inter_line_distance
        #    median_vert_dist = median_inter_line_distance


        # NOTE: cannot restrict ourselves to just looking at similar y coords, since !
        # but: could take all columns with same number of lines!
        
        # assume that most of the image has been segmented correctly
        # assume that order is approximately the same

        # NOTE: we can e.g. run linear regression or RANSAC on each line

        row_x_coord_collector = []
        row_y_coord_collector = []
        for idx in range(largest_slice_line_start_idxs_length):  # loop over rows
            row_x_coords = []
            row_y_coords = []
        
            # idea for a more robust algorithm:
            # first, "expand" to the left, always assigning to each of the elements of the latest column
            # (which is initially "largest_slice_line_start_idxs_slice_idx") the element of the current column with the 
            # smallest y distance to that element
            # then, expand to the right, doing the same thing
            # could also try to ensure that there is an injective mapping of only the best corresponding pairs

            # will contain y coordinates:
            # largest_slice_line_start_idxs = slice_line_start_idxs_collector[largest_slice_line_start_idxs_slice_idx]
            # largest_slice_line_x = slice_line_start_idxs_slice_idxs[largest_slice_line_start_idxs_slice_idx]


            for slice_idx, slice_line_start_idxs in zip(slice_line_start_idxs_slice_idxs,
                                                        slice_line_start_idxs_collector):
                if len(slice_line_start_idxs) > idx:
                    row_y_coords.append(slice_line_start_idxs[idx])
                    row_x_coords.append(slice_idx)
            row_x_coord_collector.append(row_x_coords)
            row_y_coord_collector.append(row_y_coords)

        angles_deg = []
        draw = ImageDraw.Draw(log_img)
        for row_xs, row_ys in zip(row_x_coord_collector, row_y_coord_collector):
            if len(row_xs) < 2:  # could add robustness by discarding lines with less than some threshold of samples
                continue
            reg = RANSACRegressor(base_estimator=LinearRegression(fit_intercept=True))
            reg.fit(np.expand_dims(np.array(row_xs), axis=-1), row_ys)

            slope = reg.estimator_.coef_[0]
            draw.line((0, reg.estimator_.intercept_, log_img.width, reg.estimator_.intercept_ + log_img.width * slope), fill=(255, 0, 0))
            angles_deg.append(np.arctan(slope) / np.pi * 180.0)
            

        angle_deg = np.median(angles_deg)
        
        if log_dict is not None:
            log_dict[f'lines_pre_rot{"_" + log_dict_suffix if log_dict_suffix is not None else ""}'] = log_img

        return angle_deg

def get_deg_rot_angle_hough_transform(cv2_img, log_dict=None, log_dict_suffix=None):
    # attempt to determine horizontal rotation of ECG
    # see OpenCV docs on cv::HoughLines function
    # (https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a)
    # for that, cluster on angles
    # theta: "0 ~ vertical line, pi/2 ~ horizontal line"
    # look for cluster within 45° of x-axis
    # input: cv2 img
    # output: PIL img
    lines_pre_rot = detect_lines(cv2_img)

    cv2_img_rgb = np.copy(cv2_img) if len(cv2_img.shape) == 3\
                  else np.repeat(np.expand_dims(cv2_img, axis=-1), 3, axis=-1)

    if log_dict is not None:
        # paint lines and save
        for line in lines_pre_rot:
            rho, theta = line[0], line[1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)

            cv2.line(cv2_img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 1)

        log_dict[f'lines_pre_rot{"_" + log_dict_suffix if log_dict_suffix is not None else ""}'] =\
            Image.fromarray(cv2_img_rgb)
    
    thetas_pre_rot = np.expand_dims(np.array([line[1] for line in lines_pre_rot]), axis=-1)
    k_means_pre_rot = KMeans(n_clusters=NUM_LINE_CLUSTERS).fit(thetas_pre_rot)
    hor_cluster_idx_pre_rot, hor_cluster_angle_pre_rot, hor_cluster_votes_pre_rot =\
        get_matching_cluster(k_means_pre_rot, line_range=HOR_LINE_RANGE_MEDIUM)

    return hor_cluster_angle_pre_rot / np.pi * 180.0 - 90.0

    #with Image.open(path).convert('RGB') as pre_rot_img_pil:
    #with Image.fromarray(cv2_img) as pre_rot_img_pil:
    #    rot = pre_rot_img_pil.rotate(angle=hor_cluster_angle_pre_rot / np.pi * 180.0 - 90.0, expand=False)
    #return rot

def detect_hor_vert_cell_distances_loop(straightened_cv2_img, log_dict=None, log_dict_suffix=None):
    mean_hor_dist, median_hor_dist = np.nan, np.nan
    mean_vert_dist, median_vert_dist = np.nan, np.nan

    log_img = Image.fromarray(np.copy(straightened_cv2_img) if len(straightened_cv2_img.shape) == 3\
                              else np.repeat(np.expand_dims(straightened_cv2_img, axis=-1), 3, axis=-1))

    straightened_cv2_img_height = straightened_cv2_img.shape[0]
    straightened_cv2_img_width = straightened_cv2_img.shape[1]

    # "slice" = scan column if we're looking for the vertical inter-cell distance (for horizontal lines);
    #           scan row if we're looking for the horizontal inter-cell distance (for vertical lines)
    NUM_SLICES_TO_PROBE = 10
    MODE_HOR = 0  # looking for the horizontal inter-cell distance
    MODE_VERT = 1  # looking for the vertical inter-cell distance

    STATE_OUTSIDE_LINE = 0
    STATE_INSIDE_LINE = 1
    for mode in [MODE_HOR, MODE_VERT]:  # 0: horizontal; 1: vertical
        # when looking for hor dist, detect vert lines, and vice versa
        mode_img_value_to_detect = {MODE_HOR: 85, MODE_VERT: 170}[mode]
        bg_class = 0

        probe_dim_value = straightened_cv2_img_height if mode == MODE_HOR else straightened_cv2_img_width
        scanthrough_dim_value = straightened_cv2_img_width if mode == MODE_HOR else straightened_cv2_img_height
        slices_to_scan_idxs = []
        for _slice_to_scan_idx in range(0, probe_dim_value, int(probe_dim_value / NUM_SLICES_TO_PROBE)):
            slice_to_scan_idx = _slice_to_scan_idx
            while slice_to_scan_idx in slices_to_scan_idxs and slice_to_scan_idx < probe_dim_value:
                slice_to_scan_idx += 1
            if slice_to_scan_idx < probe_dim_value:
                slices_to_scan_idxs.append(slice_to_scan_idx)

        inter_line_distances = []

        for slice_to_scan_idx in slices_to_scan_idxs:
            state = STATE_OUTSIDE_LINE
            MIN_INTER_LINE_DISTANCE = 5
            MIN_LINE_EXIT_DISTANCE = 5
            # we'll just assume that there are no/very little false positives
            # go through slice; once we find a "line pixel", switch to STATE_INSIDE_LINE; once we leave, switch to 
            # STATE_OUTSIDE_LINE; if another line has been found in the range of the last MIN_INTER_LINE_DISTANCE pixels
            # do not trigger a new line again
            # (be careful here to respect all state switches)

            # note that we might accidentally hit a whole line of the other orientation
            # -> only consider pixels of the class we are looking for
            # -> however, some patches might correspond to intersections of lines, or the ECG curve may intersect
            #    some lines
            # -> if we are *in* a line, we should only leave that line if we encounter a background pixel, not any other
            #    kind of pixel

            slice_line_start_idxs = []
            slice_last_line_element_idx = -np.inf
            for slice_element_idx in range(scanthrough_dim_value):
                pixel_x = slice_element_idx if mode == MODE_HOR else slice_to_scan_idx
                pixel_y = slice_to_scan_idx if mode == MODE_HOR else slice_element_idx
                pixel_value = straightened_cv2_img[pixel_y, pixel_x]

                # straightened_cv2_img has elements from 0 to 255;
                # if pixel_value == 85, it's a hor line; if pixel_value == 170, it's a vert line
                last_line_start = -np.inf if len(slice_line_start_idxs) == 0 else slice_line_start_idxs[-1]

                # the lines may be very thin, even just one pixel wide
                # hence, we immediately enter STATE_INSIDE_LINE whenever we encounter a white pixel

                # however, we assume that the lines have at least some minimum amount of distance between them

                if state == STATE_OUTSIDE_LINE\
                   and pixel_value == mode_img_value_to_detect\
                   and (slice_element_idx - last_line_start) >= MIN_INTER_LINE_DISTANCE:
                    state = STATE_INSIDE_LINE
                    log_img.putpixel((pixel_x, pixel_y), (0, 255, 0) if mode == MODE_HOR else (0, 0, 255))
                    slice_line_start_idxs.append(slice_element_idx)
                elif state == STATE_INSIDE_LINE\
                    and pixel_value == bg_class and slice_last_line_element_idx >= MIN_LINE_EXIT_DISTANCE:
                    # for now, we don't use the exit points of the lines for anything
                    # we just allow new lines to be detected by entering STATE_OUTSIDE_LINE again
                    state = STATE_OUTSIDE_LINE

                if state == STATE_INSIDE_LINE:
                    slice_last_line_element_idx = slice_element_idx
                    log_img.putpixel((pixel_x, pixel_y), (0, 255, 0) if mode == MODE_HOR else (0, 0, 255))

            slice_line_start_idxs_np = np.array(slice_line_start_idxs)
            inter_line_distances.append(slice_line_start_idxs_np[1:] - slice_line_start_idxs_np[:-1])

        inter_line_distances_np = np.concatenate(inter_line_distances)
        
        mean_inter_line_distance = np.mean(inter_line_distances_np)
        median_inter_line_distance = np.median(inter_line_distances_np)

        if mode == MODE_HOR:
            mean_hor_dist = mean_inter_line_distance
            median_hor_dist = median_inter_line_distance
        else:
            mean_vert_dist = mean_inter_line_distance
            median_vert_dist = median_inter_line_distance
    
    if log_dict is not None:
        log_dict[f'lines_hor_vert{"_" + log_dict_suffix if log_dict_suffix is not None else ""}'] = log_img

    return mean_hor_dist, median_hor_dist, mean_vert_dist, median_vert_dist


def detect_hor_vert_cell_distances_mean_shift(straightened_cv2_img, log_dict=None, log_dict_suffix=None):
    # after straightening, we can detect lines again, but now use a smaller range
    # then, we can determine the "horizontal lines" and "vertical lines" clusters based on the angles,
    # then merge lines with similar angle and location together (maybe can even write own basic clustering algo?)
    # then look at "distance to closest line in merged horizontal cluster", then use average of distances as
    # vertical/horizontal big cell size
    # for transcribing, some sort of continuity prior would be nice (line may disappear, reappear, etc.)
    # otherwise, could e.g. take mean (better: median?), then apply smoothing
    # need to detect start and end of time series on picture
    # could also do a DBSCAN on the curve and select the largest cluster

    # start by assigning a value on each x timestep
    # after getting the values, we need to rescale to some standard frequency
    # perform interpolation between obtained points, e.g. bilinear

    # could check how good obtained time series are compared to originals (e.g. by plotting?)

    lines_post_rot = detect_lines(straightened_cv2_img)

    
    straightened_cv2_img_rgb_1 = np.copy(straightened_cv2_img) if len(straightened_cv2_img.shape) == 3\
                                 else np.repeat(np.expand_dims(straightened_cv2_img, axis=-1), 3, axis=-1)
    
    for line in lines_post_rot:
        rho, theta = line[0], line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(straightened_cv2_img_rgb_1, (x1, y1), (x2, y2), (255, 0, 0), 1)

    log_dict[f'lines_post_rot{"_" + log_dict_suffix if log_dict_suffix is not None else ""}'] =\
        Image.fromarray(straightened_cv2_img_rgb_1)

    thetas_post_rot = np.expand_dims(np.array([line[1] for line in lines_post_rot]), axis=-1)
    k_means_post_rot = KMeans(n_clusters=NUM_LINE_CLUSTERS).fit(thetas_post_rot)

    mean_hor_dist, median_hor_dist = np.nan, np.nan
    mean_vert_dist, median_vert_dist = np.nan, np.nan

    straightened_cv2_img_rgb_2 = np.copy(straightened_cv2_img) if len(straightened_cv2_img.shape) == 3\
                                 else np.repeat(np.expand_dims(straightened_cv2_img, axis=-1), 3, axis=-1)

    MODE_HOR = 0
    MODE_VERT = 1
    for mode in [MODE_HOR, MODE_VERT]:  # 0: horizontal; 1: vertical
        # use a smaller range this time for horizontal lines
        cluster_idx_post_rot, cluster_angle_post_rot, cluster_votes_post_rot =\
            get_matching_cluster(k_means_post_rot,
                                 line_range=HOR_LINE_RANGE_SMALL if mode == MODE_HOR else VERT_LINE_RANGE_MEDIUM)
        cluster_line_idxs_post_rot = np.squeeze(np.argwhere(k_means_post_rot.labels_ == cluster_idx_post_rot))
        cluster_lines_post_rot = np.stack([lines_post_rot[idx] for idx in cluster_line_idxs_post_rot])
        cluster_lines_rhos_post_rot = [line[0] for line in lines_post_rot]
        cluster_lines_thetas_post_rot = [line[1] for line in lines_post_rot]
        # try mean-shift clustering to merge similar lines

        # first, Z-standardize the lines
        cluster_lines_rhos_mu_post_rot = np.mean(cluster_lines_rhos_post_rot)
        cluster_lines_rhos_sigma_post_rot = np.std(cluster_lines_rhos_post_rot)

        cluster_lines_thetas_mu_post_rot = np.mean(cluster_lines_thetas_post_rot)
        cluster_lines_thetas_sigma_post_rot = np.std(cluster_lines_thetas_post_rot)

        cluster_lines_standardized_post_rot =\
            np.stack([np.stack([(line[0] - cluster_lines_rhos_mu_post_rot) / cluster_lines_rhos_sigma_post_rot,
                                (line[1] - cluster_lines_thetas_mu_post_rot) / cluster_lines_thetas_sigma_post_rot])
                      for line in cluster_lines_post_rot])

        mean_shift = MeanShift(bandwidth=0.02).fit(cluster_lines_standardized_post_rot)

        # recover lines after mean shift
        cluster_lines_recovered_post_rot = \
            np.stack([np.stack([line[0] * cluster_lines_rhos_sigma_post_rot + cluster_lines_rhos_mu_post_rot,
                                line[1] * cluster_lines_thetas_sigma_post_rot + cluster_lines_thetas_mu_post_rot])
                      for line in mean_shift.cluster_centers_])

        # calculate distance between lines

        coords = []

        # paint lines and save
        for line in cluster_lines_recovered_post_rot:
            rho, theta = line[0], line[1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(straightened_cv2_img_rgb_2, (x1, y1), (x2, y2), (0,
                                                                      255 if mode == MODE_VERT else 0,
                                                                      255 if mode == MODE_HOR else 0), 1)

            if mode == MODE_HOR:
                coords.append(y0)
            else:
                coords.append(x0)

        coords_sorted_np = np.stack(sorted(coords))
        coords_distance = coords_sorted_np[1:] - coords_sorted_np[:-1]
        coords_mean_distance = np.mean(coords_distance)
        coords_median_distance = np.median(coords_distance)

        if mode == MODE_HOR:
            mean_hor_dist = coords_mean_distance
            median_hor_dist = coords_median_distance
            # print(f'Mean y distance: {coords_mean_distance}')
            # print(f'Median y distance: {coords_median_distance}')
        else:
            mean_vert_dist = coords_mean_distance
            median_vert_dist = coords_median_distance
            # print(f'Mean x distance: {coords_mean_distance}')
            # print(f'Median x distance: {coords_median_distance}')

    if log_dict is not None:
        log_dict[f'lines_rec_mean_shift{"_" + log_dict_suffix if log_dict_suffix is not None else ""}'] =\
            Image.fromarray(straightened_cv2_img_rgb_2)
    
    return mean_hor_dist, median_hor_dist, mean_vert_dist, median_vert_dist
