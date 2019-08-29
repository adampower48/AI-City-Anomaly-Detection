import PIL
import cv2 as cv
import numpy as np
import pandas as pd
import torch

import cropping
from mmdet.apis import init_detector, inference_detector
from utils import VideoReader, ImageReader
import matplotlib.pyplot as plt
import matplotlib


class Detector:
    """
    built using: https://github.com/open-mmlab/mmdetection/tree/master/configs/htc
    (HTC + DCN + ResNeXt-101-FPN, mAP=50.7 model)

    Might take some fiddling to make it work with any mmdetection model

    config_file: "htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py"
    checkpoint_file: "htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth"
    class_restrictions: list of classes to detect, others are discarded. if None, it will detect all classes

    """


    def __init__(self, config_path, model_path, verbose_interval=None, class_restrictions=None):
        torch.cuda.empty_cache()
        self.model = init_detector(config_path, model_path, device='cuda:0')
        # self.class_labels = self.model.CLASSES
        self.class_restrictions = class_restrictions

        self.verbose_interval = verbose_interval

        ###
        self.i = 0
        ###


    def detect_objects(self, img):
        """
        Runs object detection on an image.
        Returns bounding boxes [x1, y1, x2, y2], class labels, and confidence scores
        """

        results = inference_detector(self.model, img)
        if type(results) == tuple and len(results) == 2:  # if detector has segmentation head
            results, segments = results

        bbox_and_scores = np.vstack(results)
        bboxes, scores = bbox_and_scores[:, :4], bbox_and_scores[:, 4]
        labels = np.concatenate([[i] * len(bbox) for i, bbox in enumerate(results)]).astype(int)

        order = np.argsort(scores)[::-1]  # sort
        return bboxes[order], labels[order], scores[order]


    def detect_crop(self, img, crop_boxes):
        """
        Splits an image into boxes, upscales them, performs detection, downscales detections, merges detections

        img: numpy array of image
        crop_boxes: list of crop bounding boxes [x1, y1, x2, y2]

        returns: detection results [[x1, y1, x2, y2, score, class], ...]
        """
        img = img.astype(np.uint8)
        print(crop_boxes)
        pil_img = PIL.Image.fromarray(img)
        crops = cropping.crop_image(pil_img, crop_boxes)
        resized, biggest = cropping.resize_crops(crops)
        print(resized)
        resized_np = (np.array(img) for img in resized)

        ### draw crop boxes
        img2 = np.copy(img)
        for x1, y1, x2, y2 in crop_boxes:
            cv.rectangle(img2, (x1, y1), (x2, y2), (255, 255, 0), thickness=5)

        ###

        crop_results = self.detect_images(resized_np, verbose=False)

        bboxes = []
        for i, x1, y1, x2, y2, score, cls in crop_results.values:
            bboxes.append(cropping.cropped_detection_to_original((x1, y1, x2, y2), crop_boxes[int(i)], biggest))
        bboxes = np.array(bboxes)

        scores = crop_results["score"].values
        labels = crop_results["class"].values

        ### draw cropped results
        cmap = plt.get_cmap("viridis")
        for (x1, y1, x2, y2), score in zip(bboxes, scores):
            if score < 0.2:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            col = tuple(int(c * 255) for c in cmap(score)[:3])
            cv.rectangle(img2, (x1, y1), (x2, y2), col, thickness=2)

        if self.i % 10 == 0:
            plt.imshow(img2)
            plt.show()
        self.i += 1
        ###

        order = np.argsort(scores)[::-1]  # sort
        return bboxes[order], labels[order], scores[order]


    def detect_images(self, images, frames=None, crop_boxes=None, verbose=True, max_area_percent=0.5):
        """
        Runs object detection on images, yields all results at the end.

        images: iterable of numpy array images
        frames: iterable of frame numbers/names corresponding to images
        crop_boxes: if provided, will crop and rescale for detection
        verbose: override for self.verbose_interval

        Returns dataframe with detection results
        """

        results = []
        for i, img in enumerate(images):
            frame = i if frames is None else frames[i]

            if crop_boxes is not None:
                bboxes, labels, scores = self.detect_crop(img, crop_boxes)
            else:
                bboxes, labels, scores = self.detect_objects(img)

            for (x1, y1, x2, y2), cls, score in zip(bboxes, labels, scores):
                if self.class_restrictions and cls not in self.class_restrictions:
                    continue

                percent_area = (((x2 - x1) * (y2 - y1)) / (img.shape[0] * img.shape[1])) ** 0.5
                if percent_area < max_area_percent:
                    results.append([frame, x1, y1, x2, y2, score, cls])

            if verbose and self.verbose_interval and (i % self.verbose_interval) == 0:
                print(f"Detecting image: {frame}")
        #                 print(*results[-5:], sep="\n")

        return pd.DataFrame(data=results, columns=["frame", "x1", "y1", "x2", "y2", "score", "class"])


    def detect_images_generator(self, images, frames=None, crop_box_gen=None):
        """
        Runs object detection on images, yielding the results one frame at a time.

        images: iterable of numpy array images
        frames: iterable of frame numbers/names corresponding to images
        crop_box_gen: if provided, will crop and rescale for detection, must have the same length as images

        Returns frame, dataframe with detection results.
        """

        for i, img in enumerate(images):
            frame = i if frames is None else frames[i]

            if crop_box_gen is not None:
                crop_boxes = next(crop_box_gen)
                if len(crop_boxes) > 0:
                    bboxes, labels, scores = self.detect_crop(img, crop_boxes)
                else:
                    bboxes, labels, scores = self.detect_objects(img)

            else:
                bboxes, labels, scores = self.detect_objects(img)

            results = []
            for (x1, y1, x2, y2), cls, score in zip(bboxes, labels, scores):
                if self.class_restrictions and cls not in self.class_restrictions:
                    continue

                results.append([frame, x1, y1, x2, y2, score, cls])

            yield frame, pd.DataFrame(data=results, columns=["frame", "x1", "y1", "x2", "y2", "score", "class"])


    def detect_images_getter(self, image_getter, crop_boxes=None):
        """
        Returns a function that can be used to get detection results for specific frames.
        Example usage:
            results_getter = detector.detect_images_getter(vid_reader.get_frame)
            results = results_getter(100)

        :param image_getter: function used to get an image. eg VideoReader.get_frame
        :param crop_boxes: if provided, will crop and rescale for detection
        :return:
        """


        def detect(frame):
            img = image_getter(frame)

            if crop_boxes is not None:
                bboxes, labels, scores = self.detect_crop(img, crop_boxes)
            else:
                bboxes, labels, scores = self.detect_objects(img)

            results = []
            for (x1, y1, x2, y2), cls, score in zip(bboxes, labels, scores):
                if self.class_restrictions and cls not in self.class_restrictions:
                    continue

                results.append([frame, x1, y1, x2, y2, score, cls])

            return frame, pd.DataFrame(data=results, columns=["frame", "x1", "y1", "x2", "y2", "score", "class"])


        return detect


    def label_image(self, img, bboxes, labels, scores, score_thresh=0.1):
        """
        Returns an image with bounding boxes drawn on it
        """

        img = np.copy(img)

        for (x1, y1, x2, y2), cls, score in zip(bboxes, labels, scores):
            if score < score_thresh:
                continue

            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)  # draw boxes

        return img


def detect_video(video_path, output_path, model_config_path, model_checkpoint_path):
    """
    Runs object detection on a video. Saves results as a csv.
    """

    vid = VideoReader(video_path)
    images, filenames = vid.load_video()

    model = Detector(model_config_path, model_checkpoint_path)
    results = model.detect_images(images, filenames)

    results.to_csv(output_path, index=None)


def detect_image_folder(image_folder, output_path, model_config_path, model_checkpoint_path):
    """
    Runs object detection on a folder of images. Saves the results to a csv
    """

    img_reader = ImageReader(image_folder)

    model = Detector(model_config_path, model_checkpoint_path)
    results = model.detect_images(img_reader.load_images(), img_reader.filenames)

    results.to_csv(output_path, index=None)
