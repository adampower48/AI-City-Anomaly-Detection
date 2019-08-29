"""
All of the stuff with temporal-spatial matrices in here.
This is where most of the processing outside of detection is done.

"""


import numpy as np
import pandas as pd
import types

import utils
from reid.extractor import ReidExtractor
from utils import ResultsDict, VideoReader
import matplotlib.pyplot as plt
import cv2 as cv


def add_boxes(bboxes, ignore_matrix):
    """
    Creates tmp_score and tmp_detect arrays.

    bboxes: list of bounding boxes and scores [x1, y1, x2, y2, score]
    ignore_matrix: Boolean mask of region to ignore for boxes.
    """
    h, w = ignore_matrix.shape

    tmp_score = np.zeros((h, w))
    tmp_detect = np.zeros((h, w), dtype=bool)

    for x1, y1, x2, y2, score in bboxes:  # for each box
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        tmp_score[y1:y2, x1:x2] = np.maximum(score, tmp_score[y1:y2, x1:x2])  # add box
        tmp_detect[y1:y2, x1:x2] = True

    tmp_score = utils.mask(tmp_score, ignore_matrix)  # get rid of stuff in ignore regions
    tmp_detect &= ignore_matrix

    return tmp_score, tmp_detect


def get_anomalies_preprocessed(video_path, reid_model_path, frame_by_frame_results_path, static_results_path,
                               ignore_matrix_path=None, reid_model_name="resnet50", start_frame=1, frame_interval=20,
                               abnormal_duration_thresh=60, detect_thresh=5, undetect_thresh=8, score_thresh=0.3,
                               light_thresh=0.8, anomaly_score_thresh=0.7, similarity_thresh=0.95,
                               suspicious_time_thresh=18, verbose=False):
    """
    Performs the anomaly detection, assumes all the detection results, background modelling, etc is already done.

    video_path: path to raw video
    reid_model_path: path to re-ID model checkpoint
    frame_by_frame_results_path: path to object detection results on raw video
    static_results_path: path to object detection results on background images
    ignore_matrix_path: path to ignore region mask
    reid_model_name: backbone used for reid model
    start_frame: video frame to start from
    frame_interval: interval between frames to do calculations on
    abnormal_duration_thresh: duration (in seconds) to consider an object abnormal
    detect_thresh: duration (in frames) to consider an object for tracking
    undetect_thresh: duration (in frames) to stop considering an object for tracking
    score_thresh: detection score threshold for bounding boxes
    light_thresh: brightness threshold (not sure what it does)
    anomaly_score_thresh: threshold to consider an object an anomaly
    similarity_thresh: threshold for object re-ID
    suspicious_time_thresh: duration (in seconds) for an object to be considered suspicious
    verbose: verbose printing


    """

    # Read result data
    fbf_bbox_df = pd.read_csv(frame_by_frame_results_path)
    static_results_df = pd.read_csv(static_results_path)

    fbf_results_dict = ResultsDict.from_df(fbf_bbox_df)
    static_results_dict = ResultsDict.from_df(static_results_df)

    vid = VideoReader(video_path)

    return get_anomalies_sequential(vid, reid_model_path, fbf_results_dict, static_results_dict,
                                    ignore_matrix_path, reid_model_name, start_frame, frame_interval,
                                    abnormal_duration_thresh, detect_thresh, undetect_thresh, score_thresh,
                                    light_thresh, anomaly_score_thresh, similarity_thresh, suspicious_time_thresh,
                                    verbose)


def get_anomalies_sequential(video_reader, reid_model_path, fbf_results_dict, static_results_dict,
                             ignore_matrix_gen=None, reid_model_name="resnet50", start_frame=1, frame_interval=20,
                             abnormal_duration_thresh=60, detect_thresh=5, undetect_thresh=8, score_thresh=0.3,
                             light_thresh=0.8, anomaly_score_thresh=0.7, similarity_thresh=0.95,
                             suspicious_time_thresh=18, verbose=False, anomaly_nms_thresh=0.8):
    """
    Performs the anomaly detection. Sequential version

    video_reader: VideoReader object for raw video
    reid_model_path: path to re-ID model checkpoint
    fbf_results_dict: ResultsDict object for frame-by-frame/raw video detection results
    static_results_dict: ResultsDict object for static/background detection results
    ignore_matrix_gen: generator yielding ignore matrix, must have the same interval as frame_interval.
        Or single numpy array, or path to .npy file.
    reid_model_name: backbone used for reid model
    start_frame: video frame to start from
    frame_interval: interval between frames to do calculations on
    abnormal_duration_thresh: duration (in seconds) to consider an object abnormal
    detect_thresh: duration (in frames) to consider an object for tracking
    undetect_thresh: duration (in frames) to stop considering an object for tracking
    score_thresh: detection score threshold for bounding boxes
    light_thresh: brightness threshold (not sure what it does)
    anomaly_score_thresh: threshold to consider an object an anomaly
    similarity_thresh: threshold for object re-ID
    suspicious_time_thresh: duration (in seconds) for an object to be considered suspicious
    verbose: verbose printing
    anomaly_nms_thresh: IoU threshold for anomaly NMS.


    """


    def get_ignore_gen(ign_matrix):
        """
        Handles different inputs for ignore matrix

        :param ign_matrix:
        :return:
        """

        if isinstance(ign_matrix, types.GeneratorType):
            return ign_matrix

        # load/create matrix
        if ign_matrix is None:
            matrix = np.ones((h, w), dtype=bool)  # Dont ignore anything

        elif type(ign_matrix) == str:  # filename
            matrix = np.load(ign_matrix).astype(bool)

        else:
            raise TypeError("Invalid ignore matrix type:", type(ign_matrix))

        return (matrix for _ in iter(int, 1))  # infinite generator


    # Get video data
    num_frames, framerate, image_shape = video_reader.nframes, video_reader.framerate, video_reader.img_shape

    # load model
    reid_model = ReidExtractor(reid_model_name, reid_model_path)

    # Set up information matrices
    h, w, _ = image_shape

    ignore_matrix_gen = get_ignore_gen(ignore_matrix_gen)

    detect_count_matrix = np.zeros((h, w))
    undetect_count_matrix = np.zeros((h, w))
    start_time_matrix = np.zeros((h, w))
    end_time_matrix = np.zeros((h, w))
    score_matrix = np.zeros((h, w))
    state_matrix = np.zeros((h, w), dtype=bool)  # State matrix, 0/1 distinguishes suspicious candidate states

    if verbose:
        print(f"total frames: {num_frames}, framerate: {framerate}, height: {h}, width: {w}")
        print("-------------------------")

    ### Main loop
    start = False
    tmp_start = False
    all_results = []
    anomaly_now = {}
    for frame in range(start_frame, num_frames, frame_interval):
        try:
            ignore_matrix = next(ignore_matrix_gen)

            # if frame % (10*30) == 0:
            #     plt.imshow(ignore_matrix)
            #     plt.show()
        except StopIteration:
            pass  # keep same ignore matrix

        # Comment out if not using crop boxes, not needed
        # if fbf_results_dict.max_frame < static_results_dict.max_frame:
        #     fbf_results_dict.gen_next()

        # create tmp_score, tmp_detect
        static_results = static_results_dict[frame]
        if static_results is not None:
            boxes = static_results.loc[static_results["score"] > score_thresh,
                                       ["x1", "y1", "x2", "y2", "score"]].values
        else:
            boxes = []

        tmp_score, tmp_detect = add_boxes(boxes, ignore_matrix)

        ### plotting
        # img = video_reader.get_frame(frame)
        # cmap = plt.get_cmap("viridis")
        # for x1, y1, x2, y2, score in boxes:
        #     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        #     col = tuple(int(c * 255) for c in cmap(score)[:3])
        #     cv.rectangle(img, (x1, y1), (x2, y2), col, thickness=2)
        #
        # if frame % 12 == 0:
        #     plt.imshow(img)
        #     plt.show()
        ###


        if verbose:
            print(f"frame: {frame}")

            if len(boxes) > 0:
                print("\tboxes:", len(boxes))

        score_matrix += tmp_score  # add running totals
        detect_count_matrix += tmp_detect

        # Update detection matrices
        undetect_count_matrix += ~ tmp_detect
        undetect_count_matrix[tmp_detect] = 0

        # Update time matrices
        start_time_matrix[detect_count_matrix == 1] = -600 if frame == 1 else frame  # why -600 for frame 1?
        end_time_matrix[detect_count_matrix > 0] = frame

        # Update state matrices
        state_matrix[detect_count_matrix > detect_thresh] = True

        # Detect anomaly
        time_delay = utils.mask(end_time_matrix - start_time_matrix, state_matrix)
        delay_max_idx = np.unravel_index(time_delay.argmax(), time_delay.shape)

        #         print(f"\tmax delay: {time_delay.max()}, start: {start_time_matrix[delay_max_idx]}, end: {end_time_matrix[delay_max_idx]}, state: {state_matrix[delay_max_idx]}")
        if not start and time_delay.max() / framerate > abnormal_duration_thresh:  # and score_matrix[delay_max_idx]/detect_count_matrix[delay_max_idx]>0.8:

            delay_max_idx = np.unravel_index(time_delay.argmax(), time_delay.shape)

            # backtrack the start time
            time_frame = int(start_time_matrix[delay_max_idx] / 5) * 5  # + 1  # why 5s and 1?

            G = np.where(detect_count_matrix < detect_count_matrix[delay_max_idx] - 2, 0,
                         1)  # What does G represent?, why -2?
            region = utils.search_region(G, delay_max_idx)

            # vehicle reid
            if 'start_time' in anomaly_now and (time_frame / framerate - anomaly_now['end_time']) < 30:  # why 30?
                f1_frame_num = max(1, anomaly_now['start_time'] * framerate)
                f2_frame_num = max(1, time_frame)

                similarity = reid_model.similarity(video_reader.get_frame(f1_frame_num),
                                                   video_reader.get_frame(f2_frame_num),
                                                   anomaly_now["region"], region)

                if similarity > similarity_thresh:
                    time_frame = int(anomaly_now['start_time'] * framerate / 5) * 5  # + 1  # why 5s and 1?
                else:
                    anomaly_now['region'] = region

            else:
                anomaly_now['region'] = region

            # IoU stuff
            max_iou = 1
            count = 1
            start_time = time_frame
            tmp_len = 1
            raio = 1
            while (max_iou > 0.1 or tmp_len < 40 or raio > 0.6) and time_frame > 1:  # why 0.1, 40, 0.6?
                raio = count / tmp_len

                print("time frame:", time_frame)
                fbf_results = fbf_results_dict[time_frame]
                if fbf_results is not None:
                    bboxes = fbf_results[["x1", "y1", "x2", "y2", "score"]].values
                    max_iou = utils.compute_iou(anomaly_now['region'], bboxes)

                else:
                    max_iou = 0

                time_frame -= 5  # why 5?
                if max_iou > 0.3:  # why 0.3?
                    count += 1
                    if max_iou > 0.5:  # why 0.5?  # they mention 0.5 IoU in the paper for NMS, might not be this 
                        start_time = time_frame

                tmp_len += 1

            # back track start_time, until brightness at that spot falls below a threshold
            for time_frame in range(start_time, 1, -5):
                #                 print(f"\ttimeframe: {time_frame}")
                tmp_im = video_reader.get_frame(time_frame)
                if utils.compute_brightness(tmp_im[region[1]:region[3], region[0]:region[2]]) <= light_thresh:
                    break

                start_time = time_frame

            anomaly_now['start_time'] = max(0, start_time / framerate)
            anomaly_now['end_time'] = max(0, end_time_matrix[delay_max_idx] / framerate)
            start = True

        elif not tmp_start and time_delay.max() > suspicious_time_thresh * framerate:
            time_frame = start_time_matrix[delay_max_idx]

            G = np.where(detect_count_matrix < detect_count_matrix[delay_max_idx] - 2, 0, 1)  # what does G represent?
            region = utils.search_region(G, delay_max_idx)

            # vehicle reid
            if 'start_time' in anomaly_now and (time_frame / framerate - anomaly_now['end_time']) < 30:  # why 30?
                f1_frame_num = max(1, anomaly_now['start_time'] * framerate)
                f2_frame_num = max(1, time_frame)

                similarity = reid_model.similarity(video_reader.get_frame(f1_frame_num),
                                                   video_reader.get_frame(f2_frame_num),
                                                   anomaly_now["region"], region)

                if similarity > similarity_thresh:
                    time_frame = int(anomaly_now['start_time'] * framerate / 5) * 5 + 1
                    region = anomaly_now['region']

            anomaly_now['region'] = region
            anomaly_now['start_time'] = max(0, time_frame / framerate)
            anomaly_now['end_time'] = max(0, end_time_matrix[delay_max_idx] / framerate)

            tmp_start = True

        if start and time_delay.max() / framerate > abnormal_duration_thresh:

            delay_max_idx = np.unravel_index(time_delay.argmax(), time_delay.shape)

            if undetect_count_matrix[delay_max_idx] > undetect_thresh:
                anomaly_score = score_matrix[delay_max_idx] / detect_count_matrix[delay_max_idx]

                print("\t", anomaly_now, anomaly_score)
                if anomaly_score > anomaly_score_thresh:
                    anomaly_now['end_time'] = end_time_matrix[delay_max_idx] / framerate
                    anomaly_now['score'] = anomaly_score

                    all_results.append(anomaly_now)
                    anomaly_now = {}

                start = False

        elif tmp_start and time_delay.max() > suspicious_time_thresh * framerate:
            if undetect_count_matrix[delay_max_idx] > undetect_thresh:

                anomaly_score = score_matrix[delay_max_idx] / detect_count_matrix[delay_max_idx]
                if anomaly_score > anomaly_score_thresh:
                    anomaly_now['end_time'] = end_time_matrix[delay_max_idx] / framerate
                    anomaly_now['score'] = anomaly_score

                tmp_start = False

        # undetect matrix change state_matrix
        state_matrix[undetect_count_matrix > undetect_thresh] = False
        undetect_count_matrix[undetect_count_matrix > undetect_thresh] = 0

        # update matrix
        tmp_detect |= state_matrix
        detect_count_matrix = utils.mask(detect_count_matrix, tmp_detect)
        score_matrix = utils.mask(score_matrix, tmp_detect)

    # Add all anomalies to the results list
    print("---", start, time_delay.max(), score_matrix[delay_max_idx], detect_count_matrix[delay_max_idx])
    if start and time_delay.max() > abnormal_duration_thresh * framerate:
        anomaly_score = score_matrix[delay_max_idx] / detect_count_matrix[delay_max_idx]
        if anomaly_score > anomaly_score_thresh:
            anomaly_now['end_time'] = end_time_matrix[delay_max_idx] / framerate
            anomaly_now['score'] = anomaly_score

            all_results.append(anomaly_now)
            anomaly_now = {}
            start = False

    # Apply Non-Maximal Supression to the results
    if all_results:
        nms_out = utils.anomaly_nms(all_results, anomaly_nms_thresh)

        #         final_result = {'start_time': 892, 'score': 0} # why 892?
        #         for nms_start_time, nms_end_time in nms_out[:, 5:7]:
        #             if nms_start_time < final_result["start_time"]:
        #                 final_result["start_time"] = max(0, int(nms_start_time - 1))
        #                 final_result["score"] = 1
        #                 final_result["end_time"] = nms_end_time

        final_results = pd.DataFrame(nms_out, columns=["x1", "y1", "x2", "y2", "score", "start_time", "end_time"])

        return final_results

    return None
