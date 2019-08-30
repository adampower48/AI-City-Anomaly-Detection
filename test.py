from pipeline import process_folder


video_dir = "/data/aicity/train"
anomaly_results_dir = "/data/aicity/winner_team/anomaly_results/train"

# todo: create all these in a temp directory
# only needed if processing by step
static_dir = "/data/aicity/winner_team/background_images/train"
frame_by_frame_results_dir = "/data/aicity/winner_team/detection_results/train_framebyframe"
static_results_dir = "/data/aicity/winner_team/detection_results/train_static"
crop_results_dir = "/data/aicity/winner_team/detection_results/train_crop"
crop_boxes_dir = "/data/aicity/winner_team/crop_boxes/train"
ignore_mask_dir = "/data/aicity/winner_team/detection_results/train_seg_masks"

# model used for tracking
reid_model_backbone = "resnet50"
reid_model_path = "/data/modules/AICity2019_winner/models/reid/reid.pth"

## DETECTOR Models
htc_config_path = "/data/modules/mmdetection/configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py"
htc_model_path = "/data/modules/mmdetection/checkpoints/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth"

# coco
# ssd_config_path = "/data/modules/mmdetection/configs/ssd512_coco_custom.py"
# ssd_model_path = "/data/modules/mmdetection/work_dirs/ssd512_coco/latest.pth"

# voc
# ssd_config_path = "/data/modules/mmdetection/configs/pascal_voc/ssd512_voc.py"
# ssd_model_path = "/data/modules/mmdetection/checkpoints/ssd512_voc_vgg16_caffe_240e_20190501-ff194be1.pth"
# ssd_model_path = "/data/modules/mmdetection/work_dirs/ssd512_voc/latest.pth"

# 1 class
# ssd_config_path = "/data/modules/mmdetection/configs/pascal_voc/ssd512_voc_1cls.py"
# ssd_model_path = "/data/modules/mmdetection/work_dirs/ssd512_voc_1cls/latest.pth"

# 2 class
ssd_config_path = "/data/modules/mmdetection/configs/pascal_voc/ssd512_voc_2cls.py"
ssd_model_path = "/data/modules/mmdetection/work_dirs/ssd512_voc_2cls/latest.pth"

# cascade_dcn_config_path = "/data/modules/mmdetection/configs/dcn/cascade_rcnn_dconv_2cls.py"
# cascade_dcn_model_path = "/data/modules/mmdetection/work_dirs/cascade_rcnn_dconv_2cls/latest.pth"

cascade_dcn_config_path = "/data/modules/mmdetection/configs/dcn/cascade_rcnn_dconv_custom.py"
cascade_dcn_model_path = "/data/modules/mmdetection/work_dirs/cascade_rcnn_dconv/latest.pth"

faster_rcnn_config_path = "./mmdet_configs/faster_rcnn_dconv_c3-5_r50_fpn_x1_shubai.py"
faster_rcnn_model_path = "/data/modules/mmdetection/checkpoints/faster_rcnn_dconv_c3-5_r50_fpn_x1_shubai.pth"

cascade_hrnet_config_path = "/data/modules/mmdetection/configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e_custom.py"
cascade_hrnet_model_path = "/data/modules/mmdetection/work_dirs/cascade_rcnn_hrnetv2p_w32/latest.pth"
##


# ==================================================================================================================

# Single video, process by step
# anomalies = full_run_single(video_id, video_dir, static_dir, frame_by_frame_results_dir, static_results_dir, crop_boxes_dir,
#                             ignore_mask_dir, detector_config_path, detector_model_path, reid_model_path, reid_model_backbone,
#                             crop_results_dir, anomaly_results_dir, ignore_area_thresh=500, anomaly_thresh=0.5)

# Single video, sequential
# anomalies = full_run_sequential(video_id, video_dir, detector_config_path, detector_model_path, reid_model_path,
#                                 reid_model_backbone, anomaly_results_dir, ignore_area_thresh=500, anomaly_thresh=0.5,
#                                 bg_interval=20)

# Folder of videos
anomalies, anomaly_times = process_folder(video_dir, static_dir, frame_by_frame_results_dir, static_results_dir,
                                          crop_boxes_dir,
                                          ignore_mask_dir, cascade_dcn_config_path, cascade_dcn_model_path, reid_model_path,
                                          reid_model_backbone,
                                          crop_results_dir, anomaly_results_dir, bg_interval=4, crop_min_obj_size=8)

for vid in anomalies:
    print(f"---------- VIDEO {vid} -------------")
    print(anomalies[vid], anomaly_times[vid])

