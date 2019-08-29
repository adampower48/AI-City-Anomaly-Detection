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


# Code Structure
- For each video: (pipeline.py)
    1. Background modelling: Filters out moving vehicles (background.py)
    2. Load detector (detection.py)
    3. Run vehicle detection on raw video (detection.py)
    4. Run vehicle detection on background (detection.py)
    5. Create perspective cropping boxes (optional) (cropping.py)
    6. Create ignore region (not implemented properly) (ignore.py)
    7. Perform anomaly detection (anomaly.py)
        1. Load vehicle re-ID model (reid/extractor.py)
        2. Initialise global spatial-temporal info matrices
        3. For each background frame:
            1. Create temp score & detect matrices, update global matrices
            2. Check for anomaly region, if there is one, backtrack to find start time
            3. Update anomaly status
        4. Finish any current anomalies & return results
    8. Combine overlapping anomalies (utils.py)
    9. Save results
    
# Problems
I had to make some small changes to the mmdetection code to make training/testing detectors work.  
The standard library should work for testing the whole code though.

~~I couldn't get cascade/faster rcnn working properly, after training they produced no detection results.~~
- I think the cause of this could have been a mix up between coco/voc labels in the validation set. 
I fixed it and got some results for cascade.