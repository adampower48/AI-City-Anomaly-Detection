import os
import pandas as pd

from anomaly import get_anomalies_preprocessed, get_anomalies_sequential
from background import calc_bg_full_video, calc_bg_tensor, calc_background
from cropping import create_crop_boxes, crop_box_generator
from detection import Detector
from ignore import create_ignore_mask, create_ignore_mask_generator
from utils import get_overlapping_time, ResultsDict, VideoReader, VideoReaderQueue, ImageReader
import numpy as np


def full_run_single(video_id, video_dir, static_dir, frame_by_frame_results_dir, static_results_dir, crop_boxes_dir,
                    ignore_mask_dir, detector_config_path, detector_model_path, reid_model_path, reid_model_backbone,
                    crop_results_dir, anomaly_results_dir,
                    bg_interval=4, bg_alpha=0.05, bg_start_frame=1, bg_threshold=5, raw_detect_interval=30,
                    crop_min_obj_size=8, crop_row_capacity=3, crop_box_aspect_ratio=2,
                    ignore_count_thresh=0.08, ignore_area_thresh=2000, ignore_score_thresh=0.1, ignore_gau_sigma=3,
                    abnormal_duration_thresh=60, detect_duration_thresh=6, undetect_duration_thresh=8,
                    bbox_score_thresh=0.3,
                    light_thresh=0.8, anomaly_thresh=0.8, similarity_thresh=0.95, suspicious_duration_thresh=18,
                    detector_verbose_interval=20, verbose=True):
    """
    Runs the full anomaly detection pipeline on a video

    video_id: video id/name
    video_dir: folder the video is in
    static_dir: folder to put the background images in
    frame_by_frame_results_dir: folder to put the raw video detection results in
    static_results_dir: folder to put the background image detection results in
    crop_boxes_dir: folder to put the crop boxes in
    ignore_mask_dir: folder to put the ignore region mask in

    detector_config_path: path to detector configuration file
    detector_model_path: path to detector model checkpoint
    reid_model_path: path to re-ID model checkpoint
    reid_model_backbone: re-ID model backbone. eg. "resnet50"

    bg_interval, bg_alpha, bg_start_frame, bg_threshold: see calc_bg_full_video function
    raw_detect_interval: number of frames between detection on raw video
    crop_min_obj_size, crop_row_capacity, crop_box_aspect_ratio: see create_crop_boxes function
    ignore_count_thresh, ignore_area_thresh, ignore_score_thresh, ignore_gau_sigma: see create_ignore_mask function
    abnormal_duration_thresh, detect_duration_thresh, undetect_duration_thresh, bbox_score_thresh,
        light_thresh, anomaly_thresh, similarity_thresh, suspicious_duration_thresh:
            See get_anomalies function

    detector_verbose_interval: detector progress printing interval
    verbose: verbose printing


    """

    # Set up file paths
    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    static_images_folder = os.path.join(static_dir, f"{video_id}")
    fbf_results_path = os.path.join(frame_by_frame_results_dir, f"{video_id}.csv")
    static_results_path = os.path.join(static_results_dir, f"{video_id}.csv")
    crop_boxes_path = os.path.join(crop_boxes_dir, f"{video_id}.csv")
    crop_results_path = os.path.join(crop_results_dir, f"{video_id}.csv")
    ignore_mask_path = os.path.join(ignore_mask_dir, f"{video_id}.npy")
    anomaly_results_path = os.path.join(anomaly_results_dir, f"{video_id}.csv")

    # Create folders
    os.makedirs(static_images_folder, exist_ok=True)
    os.makedirs(frame_by_frame_results_dir, exist_ok=True)
    os.makedirs(static_results_dir, exist_ok=True)
    os.makedirs(crop_boxes_dir, exist_ok=True)
    os.makedirs(crop_results_dir, exist_ok=True)
    os.makedirs(ignore_mask_dir, exist_ok=True)
    os.makedirs(anomaly_results_dir, exist_ok=True)

    # Read Video
    raw_video = VideoReader(video_path)

    # bg modeling
    print("Creating background...")
    calc_bg_full_video(video_path, static_images_folder, bg_interval, bg_alpha, bg_start_frame, bg_threshold, verbose)

    # Detection
    detector = Detector(detector_config_path, detector_model_path, detector_verbose_interval, class_restrictions=None)
    # class_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    #                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    #                'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    #                'tvmonitor')
    # detector.model.CLASSES = class_names
    # detector.class_labels = class_names
    ## Raw Video
    print("Detecting raw video...")
    raw_images, raw_frame_nums = raw_video.load_video(raw_detect_interval)
    fbf_results = detector.detect_images(raw_images, raw_frame_nums)
    fbf_results.to_csv(fbf_results_path, index=False)

    ## Static Images
    static_reader = ImageReader(static_images_folder)
    static_frame_names = list(map(lambda f: int(f[:-4]), static_reader.filenames))  # "123.jpg" -> 123

    print("Detecting background...")
    static_results = detector.detect_images(static_reader.load_images(), static_frame_names)
    static_results.to_csv(static_results_path, index=False)

    # Perspective Cropping
    print("Creating crop boxes...")
    create_crop_boxes(fbf_results_path, crop_boxes_path, raw_video.img_shape, crop_min_obj_size, crop_row_capacity,
                      crop_box_aspect_ratio)  # either static/fbf results should work

    # Should be able to use this in place of normal static images. Doesnt look feasable atm, way too long detection time
    crop_boxes = pd.read_csv(crop_boxes_path).values
    print("Detecting cropped background...")
    crop_detect_results = detector.detect_images(static_reader.load_images(), static_frame_names, crop_boxes=crop_boxes)
    crop_detect_results.to_csv(crop_results_path)

    #     # Ignore Region
    print("Creating ingore mask...")
    create_ignore_mask(fbf_results_path, ignore_mask_path, raw_video.img_shape, ignore_count_thresh, ignore_area_thresh,
                       ignore_score_thresh, ignore_gau_sigma)

    # Detect anomalies
    print("Detecting anomalies...")
    anomalies = get_anomalies_preprocessed(video_path, reid_model_path, fbf_results_path, static_results_path,
                                           ignore_mask_path,
                                           reid_model_backbone, bg_start_frame, bg_interval, abnormal_duration_thresh,
                                           detect_duration_thresh,
                                           undetect_duration_thresh, bbox_score_thresh, light_thresh, anomaly_thresh,
                                           similarity_thresh, suspicious_duration_thresh, verbose)

    if anomalies is not None:
        anomaly_event_times = get_overlapping_time(anomalies)

        # Save results
        print("Saving Results...")
        anomalies.to_csv(anomaly_results_path, index=False)

        return anomalies, anomaly_event_times

    else:
        return [], []


def full_run_sequential(video_id, video_dir, detector_config_path, detector_model_path, reid_model_path,
                        reid_model_backbone, anomaly_results_dir, bg_interval=4, bg_alpha=0.05, bg_start_frame=0,
                        bg_threshold=5, raw_detect_interval=30, ignore_count_thresh=0.08, ignore_area_thresh=2000,
                        ignore_score_thresh=0.1, ignore_gau_sigma=3, abnormal_duration_thresh=60,
                        detect_duration_thresh=6, undetect_duration_thresh=8, bbox_score_thresh=0.3, light_thresh=0.8,
                        anomaly_thresh=0.8, similarity_thresh=0.95, suspicious_duration_thresh=18,
                        detector_verbose_interval=20, verbose=True, crop_min_obj_size=8, crop_row_capacity=3,
                        crop_box_aspect_ratio=2):
    """
    Full run but runs one frame at a time. This should be more suitable for live processing.
    Doesnt save any intermediate calculations.

    See full_run_single for parameter info

    """

    # Set up file paths
    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    anomaly_results_path = os.path.join(anomaly_results_dir, f"{video_id}.csv")

    # Create folders
    os.makedirs(anomaly_results_dir, exist_ok=True)

    # Read Video
    raw_video = VideoReaderQueue(video_path, queue_size=8)

    # bg modeling
    print("Creating background...")
    bg_images = calc_background(raw_video.load_video()[0], bg_interval, bg_alpha, bg_start_frame, bg_threshold)  # cpu
    # bg_images = calc_bg_tensor(raw_video.load_video()[0], bg_interval, bg_alpha, bg_start_frame, bg_threshold)  # gpu, doesnt seem to speed up much
    bg_images = (img for img, _ in bg_images)  # throw out frame

    # Detection
    detector = Detector(detector_config_path, detector_model_path, detector_verbose_interval,
                        # Change this parameter depending what classes you need
                        # class_restrictions={5, 6, 13, 18},  # voc vehicles
                        class_restrictions={1, 2, 3, 5, 6, 7},  # coco vehicles
                        # class_restrictions={0}, # binary (0=vehicle, 1=not vehicle)
                        # class_restrictions=range(6),  # only vehicles (0-5=vehicles, 6=not_vehicle)
                        # class_restrictions=None  # use all classes
                        )
    print(detector.model)
    ## Raw Video
    print("Detecting raw video...")
    raw_images, raw_frame_nums = raw_video.load_video(raw_detect_interval)
    fbf_results_gen = detector.detect_images_generator(raw_images, raw_frame_nums)
    fbf_results_getter = detector.detect_images_getter(raw_video.get_frame)
    fbf_results = ResultsDict(fbf_results_gen, results_getter=fbf_results_getter, name="fbf")

    print("Detecting background...")
    static_results_gen = detector.detect_images_generator(bg_images,
                                                          range(bg_start_frame, raw_video.nframes, bg_interval))
    static_results = ResultsDict(static_results_gen, name="static")

    # Comment out this whole block if you dont want to use crop boxes
    print("Creating crop boxes")
    # crop_box_gen = crop_box_generator(fbf_results, raw_video.img_shape)
    # cropped_results_gen = detector.detect_images_generator(bg_images,
    #                                                        range(bg_start_frame, raw_video.nframes, bg_interval),
    #                                                        crop_box_gen)
    # static_results.results_gen = cropped_results_gen

    # Ignore Region
    print("Creating ingore mask...")

    # todo:
    #   I cant find a good way to create the ignore mask for live processing.
    #   It may need to not be used until a sufficient number of frames have been processed.
    #   It also isn't very quick at taking new regions into account. eg a car that stops on the grass wont get added for a while.

    # ignore_alpha = 0.1
    # ignore_alpha_2 = 1 - (1 - ignore_alpha) ** bg_interval  # adjusted for different intervals
    # ignore_mask_gen = create_ignore_mask_generator(static_results.iterator(), raw_video.img_shape, ignore_count_thresh,
    #                                                ignore_area_thresh, ignore_score_thresh, ignore_gau_sigma,
    #                                                alpha=ignore_alpha_2)
    ignore_mask_gen = None  # dont ignore anything

    anomalies = get_anomalies_sequential(raw_video, reid_model_path, fbf_results, static_results, ignore_mask_gen,
                                         reid_model_backbone, bg_start_frame, bg_interval, abnormal_duration_thresh,
                                         detect_duration_thresh,
                                         undetect_duration_thresh, bbox_score_thresh, light_thresh, anomaly_thresh,
                                         similarity_thresh, suspicious_duration_thresh, verbose)

    if anomalies is not None:
        anomaly_event_times = get_overlapping_time(anomalies)

        # Save results
        print("Saving Results...")
        anomalies.to_csv(anomaly_results_path, index=False)

        return anomalies, anomaly_event_times

    else:
        return [], []


def process_folder(video_dir, static_dir, frame_by_frame_results_dir, static_results_dir, crop_boxes_dir,
                   ignore_mask_dir, detector_config_path, detector_model_path, reid_model_path, reid_model_backbone,
                   crop_results_dir, anomaly_results_dir,
                   bg_interval=4, bg_alpha=0.05, bg_start_frame=0, bg_threshold=5, raw_detect_interval=30,
                   crop_min_obj_size=8, crop_row_capacity=3, crop_box_aspect_ratio=2,
                   ignore_count_thresh=0.08, ignore_area_thresh=2000, ignore_score_thresh=0.1, ignore_gau_sigma=3,
                   abnormal_duration_thresh=60, detect_duration_thresh=6, undetect_duration_thresh=8,
                   bbox_score_thresh=0.3,
                   light_thresh=0.8, anomaly_thresh=0.8, similarity_thresh=0.95, suspicious_duration_thresh=18,
                   detector_verbose_interval=20, verbose=True):
    """
    Processes a folder of videos

    See full_run_single function for documentation.
    """

    anomalies_dict, anomaly_times_dict = {}, {}
    for filename in sorted(os.listdir(video_dir), key=lambda f: int(f[:-4])):
        video_id = int(filename[:-4])  # "123.mp4" -> 123
        print("Processing video:", video_id)

        if video_id == 33:
            continue

        # Sequential by processing steps
        #         anomalies, anomaly_event_times = full_run_single(video_id, video_dir, static_dir, frame_by_frame_results_dir, static_results_dir, crop_boxes_dir,
        #                     ignore_mask_dir, detector_config_path, detector_model_path, reid_model_path, reid_model_backbone,
        #                     crop_results_dir, anomaly_results_dir,
        #                     bg_interval, bg_alpha, bg_start_frame, bg_threshold, raw_detect_interval,
        #                     crop_min_obj_size, crop_row_capacity, crop_box_aspect_ratio,
        #                     ignore_count_thresh, ignore_area_thresh, ignore_score_thresh, ignore_gau_sigma,
        #                     abnormal_duration_thresh, detect_duration_thresh, undetect_duration_thresh, bbox_score_thresh,
        #                     light_thresh, anomaly_thresh, similarity_thresh, suspicious_duration_thresh,
        #                     detector_verbose_interval, verbose)

        # Sequential by frame
        anomalies, anomaly_event_times = full_run_sequential(video_id, video_dir, detector_config_path,
                                                             detector_model_path,
                                                             reid_model_path, reid_model_backbone,
                                                             anomaly_results_dir,
                                                             bg_interval, bg_alpha, bg_start_frame, bg_threshold,
                                                             raw_detect_interval,
                                                             ignore_count_thresh, ignore_area_thresh,
                                                             ignore_score_thresh, ignore_gau_sigma,
                                                             abnormal_duration_thresh, detect_duration_thresh,
                                                             undetect_duration_thresh, bbox_score_thresh,
                                                             light_thresh, anomaly_thresh, similarity_thresh,
                                                             suspicious_duration_thresh,
                                                             detector_verbose_interval, verbose, crop_min_obj_size,
                                                             crop_row_capacity, crop_box_aspect_ratio)

        anomalies_dict[video_id] = anomalies
        anomaly_times_dict[video_id] = anomaly_event_times

        print(video_id, anomalies, anomaly_event_times)

    return anomalies_dict, anomaly_times_dict
