Adapted from: https://github.com/ShuaiBai623/AI-City-Anomaly-Detection

# Usage
Change the strings in test.py to fit your needs.  
Run test.py

# Using different models
All models are from the mmdetection framework. (https://github.com/open-mmlab/mmdetection)  
To use another model:
 - Add the paths to its config file and checkpoint file in test.py, and pass them to the process_folder function.
    - Config files I used can be found in mmdet_configs/
    - Pretrained models were taken from [here](https://github.com/open-mmlab/mmdetection/blob/master/docs/MODEL_ZOO.md)
    and for any changes to the network architecture, pretrained weights were copied over using the code in test_model_swap.py
 - Change the class_restrictions parameter passed to the detector in pipeline.py if needed. 
 This restricts the detector to outputting bounding boxes for those classes only.  
 
See detector_results.txt for rough performances for the models I tested.

I had to make some small changes to the mmdetection code to make training/testing detectors work.  
The standard library should work for testing the whole code though.

~~I couldn't get cascade/faster rcnn working properly, after training they produced no detection results.~~
- I think the cause of this could have been a mix up between coco/voc labels in the validation set. 
I fixed it and got some results for cascade.
