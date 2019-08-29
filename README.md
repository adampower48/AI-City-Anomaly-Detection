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

# Other notes
+ Video with gaps in them, (eg frames are black, no data) that happen during an anomaly, will create 2 separate anomaly events, or cause it to stop being tracked properly.
    + see test vid 1: 4:26 - 4:28, 6:04 - 6:06, 11:21 - 11:23, 12:58 - 13:00
    + Increasing interval between frames seems to help deal with these gaps
+ Produces about 680MB of intermediate data per 15 min video processed, mostly in background images.
    + These are not used after object detection, so the anomaly detection part can be re-ran with different hyperparameters without them.
+ Currently the detector was trained on the COCO dataset. The detector in the paper was trained on UA-DETRAC and VisDrone, with a gaussian blur applied. It should be fine tuned on these datasets.
+ Much of the code on the paper's github just does not work. Most of it is full of errors, and does not reflect the algorithm in the paper. 
+ test vid 11: doesnt pick up on stopped car, but does when the repair van comes. Seems to work even with large camera movements. Seems to detect anomaly when brightness is increased. see 7:22
+ test vid 6: Seems to be detecting the car fine, but there are periods of large drops in detection scores (see frames 10600-11000)
+ Most of the time, the anomalies happen in the ignored area, so they are not picked up.
    + Reducing the ignore_area_thresh and ignore_score_thresh parameters should help this.
    + There is also the issue of moving anomalies. eg cars swerving off the road/out of camera view.
+ Increasing the interval between detecting frames does not seem to impact performance significantly.
    + Perhaps some sort of adaptive or 2 step detection would work. Run once with a large interval to produce candidate times, then go back with a finer interval to confirm.
+ I want to write the code to run frame by frame, instead of one processing step at a time. This is needed if I want to run the program in a live setting.
+ Background creation is significantly slow. 
    + Using SSD detecting every 30 frames: 25 fps, every 4 frames: 19 fps, every 600 frames: 27 fps.
    + Background takes 37ms per frame, detection ~67ms per frame.
    + Potential solution: change bg modelling so it only calculates every x frames, instead of calulating every frame and yielding every x frames.
    + Problem was actually a bottleneck in reading images
        + Changed VideoReader to work on a separate thread, and only decode images that it actually needs.
        + Sped up process by ~4x
    + Moving calulations to GPU was actually slower, running on CPU was significantly faster
+ Using HTC model is ~450ms per frame. A 15 min video takes ~55mins to process at bg_interval=4, ~9mins at bg_interval=30