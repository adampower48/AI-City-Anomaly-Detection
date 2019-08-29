"""
Changing the last layer of classifier from COCO (80+1) or VOC (20+1) to binary (2+1)

"""

from mmdet.apis import init_detector, inference_detector
from collections import OrderedDict
import numpy as np
from detection import Detector
import torch
from mmcv.runner import save_checkpoint

# old_config_path = "/data/modules/mmdetection/configs/pascal_voc/ssd512_voc_1cls.py"
# old_checkpoint_path = "/data/modules/mmdetection/work_dirs/ssd512_voc_1cls/ssd_epoch8_1cls.pth"

old_config_path = "/data/modules/mmdetection/configs/dcn/cascade_rcnn_dconv_2cls.py"
old_checkpoint_path = "/data/modules/mmdetection/work_dirs/cascade_rcnn_dconv_2cls/baseline.pth"

# new_config_path = "/data/modules/mmdetection/configs/pascal_voc/ssd512_voc_2cls.py"
# new_checkpoint_path = "/data/modules/mmdetection/work_dirs/ssd512_voc_2cls/ssd_epoch8_2cls.pth"

new_config_path = "/data/modules/mmdetection/configs/dcn/cascade_rcnn_dconv_7cls.py"
new_checkpoint_path = "/data/modules/mmdetection/work_dirs/cascade_rcnn_dconv_7cls/baseline.pth"

old_model = init_detector(old_config_path, old_checkpoint_path, device='cuda:0')
# old_detector = Detector(old_config_path, old_checkpoint_path, class_restrictions=None)
# old_model = old_detector.model
# new_detector = Detector(new_config_path, None, class_restrictions=None)
new_model = init_detector(new_config_path, device='cuda:0')
# new_model = new_detector.model

print(old_model)
# print(new_model)
sd = OrderedDict()


for k, v in old_model.state_dict().items():
    if "bbox_head" not in k:  # dont copy last layer
        sd[k] = v

print(new_model)

new_model.load_state_dict(sd, strict=False)
save_checkpoint(new_model, new_checkpoint_path)



# res = new_detector.detect_objects(np.zeros((400, 800, 3)))

res = list(inference_detector(new_model, [np.zeros((400, 800, 3))]))
print(res)
print(len(res[0]))