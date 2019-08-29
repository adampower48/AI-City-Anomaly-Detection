import numpy as np
import pandas as pd
import skimage.measure
from scipy.ndimage import filters
import matplotlib.pyplot as plt


def combine_boxes(fbf_bbox_df, img_height, img_width, score_threshold=0.1, normalize=True):
    count_matrix = np.zeros((img_height, img_width))
    for frame, df in fbf_bbox_df.groupby("frame"):
        tmp_score = np.zeros((img_height, img_width))

        for x1, y1, x2, y2, score in df[["x1", "y1", "x2", "y2", "score"]].values:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            if score > score_threshold:
                tmp_score[y1:y2, x1:x2] = np.maximum(score, tmp_score[y1:y2, x1:x2])  # add all the boxes into one image

        count_matrix += tmp_score

    if normalize:
        # scale to [0, 1]
        count_matrix = (count_matrix - count_matrix.min()) / (count_matrix.max() - count_matrix.min())

    return count_matrix


def get_connected_regions(mask, area_threshold=2000, ):
    regions = skimage.measure.label(mask, connectivity=1)  # get connected regions

    for region_idx in np.unique(regions):
        if region_idx == 0:  # 0 is background
            continue

        region_mask = regions == region_idx
        if region_mask.sum() < area_threshold:  # get rid of small regions
            mask = np.where(region_mask, False, mask)

    return mask


def create_ignore_mask(frame_by_frame_path, ignore_matrix_path, img_shape, count_threshold=0.08, area_threshold=2000,
                       score_threshold=0.1, gaussian_sigma=3):
    # Read in bboxes
    fbf_bbox_df = pd.read_csv(frame_by_frame_path)

    # Combine bboxes
    heatmap = combine_boxes(fbf_bbox_df, img_shape[0], img_shape[1], score_threshold)

    # Create ignore mask
    mask = heatmap > count_threshold
    mask = get_connected_regions(mask, area_threshold)
    mask = filters.gaussian_filter(mask.astype(float), gaussian_sigma) > count_threshold

    # Save ignore mask
    np.save(ignore_matrix_path, mask)


def create_ignore_mask_generator(fbf_results, img_shape, count_threshold=0.08, area_threshold=2000, score_threshold=0.1,
                                 gaussian_sigma=3, alpha=0.1):
    """
    Creates a rolling ignore mask, suitable for live processing.
    """

    running_heatmap = np.zeros(img_shape[:2])
    for i, (frame, results) in enumerate(fbf_results):
        # # Maybe have an alpha like how background images are made?
        # running_heatmap = combine_boxes(results, img_shape[0], img_shape[1], score_threshold,
        #                                 normalize=False) * alpha + (1 - alpha) * running_heatmap
        # Regular average over all frames: (a + b * n) / (n + 1)
        running_heatmap = (combine_boxes(results, img_shape[0], img_shape[1], score_threshold,
                                         normalize=False) + running_heatmap * i) / (i + 1)

        # plt.imshow(running_heatmap)
        # plt.show()

        # Normalise after adding instead of before
        hm_max, hm_min = running_heatmap.max(), running_heatmap.min()

        if hm_max == hm_min:  # dont divide by 0
            heatmap_norm = running_heatmap - running_heatmap.min()
        else:
            heatmap_norm = (running_heatmap - running_heatmap.min()) / (running_heatmap.max() - running_heatmap.min())

        # Create ignore mask
        mask = heatmap_norm > count_threshold
        mask = get_connected_regions(mask, area_threshold)
        mask = filters.gaussian_filter(mask.astype(float), gaussian_sigma) > count_threshold

        yield mask
