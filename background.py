import cv2 as cv
import numpy as np
import os
import torch

from utils import VideoReader


def calc_background(images, interval=20, alpha=0.1, start_frame=1, threshold=5):
    """
    Calculates the background over all of the images by averaging them.

    images: iterable of numpy images
    interval: number of images between each background calculation
    alpha: weighting of averaging. high = more of new frame, low = more of running average.
    start_frame: frame number to start averaging on
    threshold: Mean Absolute Error threshold between frames. Only calculates if there is a significant difference.


    """

    running_bg = None
    prev_img = None
    for i, img in enumerate(images):
        if running_bg is None:  # initial image
            running_bg = img
            prev_img = img
            continue

        if i < start_frame or (i - start_frame) % interval != 0:  # every (i * internal_frame + start_frame) frames, do the calcs
            continue

        diff = np.mean(np.abs(prev_img - img))
        if diff > threshold:  # if new image is significantly different from old
            running_bg = (1 - alpha) * running_bg + alpha * img  # new background
            yield running_bg, i

        else:
            yield running_bg * 0, i  # black image

        prev_img = img


def calc_bg_tensor(images, interval=20, alpha=0.1, start_frame=1, threshold=5):
    """
    Same as calc_background function, uses GPU instead.
    Seems to be equal to or slower than normal one
    """

    running_bg = None
    prev_img = None
    for i, img in enumerate(images):
        img = torch.as_tensor(img, device="cuda", dtype=torch.float16)

        if running_bg is None:  # initial image
            running_bg = img.clone()
            prev_img = img.clone()
            continue

        if (i - start_frame) % interval != 0:  # every (i * internal_frame + start_frame) frames, do the calcs
            continue

        diff = prev_img.sub_(img).abs_().mean()
        if diff.item() > threshold:  # if new image is significantly different from old
            running_bg.mul_(1 - alpha).add_(img.mul(alpha))  # new background
            yield running_bg.to(dtype=torch.uint8, device="cpu", non_blocking=True).numpy(), i

        else:
            yield running_bg.mul(0).to(dtype=torch.uint8, device="cpu", non_blocking=True).numpy(), i  # black image

        prev_img = img


def calc_bg_full_video(video_path, output_folder, interval=20, alpha=0.1, start_frame=1, threshold=5, verbose=False):
    """
    Create background images for a single video. Assumes output_folder exists already.

    video_path: path to raw video
    output_folder: folder to put background images in
    interval, alpha, start_frame, threshold: see calc_background function
    verbose: print out progress

    """

    vid = VideoReader(video_path)
    raw_imgs, _ = vid.load_video()
    bg_images = calc_background(raw_imgs, interval, alpha, start_frame, threshold)

    for bg_img, frame in bg_images:
        filename = os.path.join(output_folder, f"{frame}.jpg")
        cv.imwrite(filename, bg_img)

        if verbose:
            print(f"{frame}/{vid.nframes}")
