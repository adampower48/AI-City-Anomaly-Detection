import background
import utils
import os
import cv2 as cv
import numpy as np

raw_video_dir = "/data/aicity/train"
negative_video_ids = set(range(1, 101, 1)) - {2, 9, 11, 14, 33, 35, 49, 51, 58, 63, 66, 72, 73, 74, 83, 91, 93, 95, 97}
bg_video_dir = "/data/aicity/train_negatives"
start_frame = 1000
framerate = 30

# for filename in sorted(os.listdir(raw_video_dir)):
#     vid_id = int(filename[:-4])
#
#     if vid_id not in negative_video_ids:
#         continue
#
#     print("Video:", vid_id)
#
#     vid_path = os.path.join(raw_video_dir, filename)
#
#     vid_reader = utils.VideoReaderQueue(vid_path, queue_size=4)
#     vid_gen, frames = vid_reader.load_video()
#     bg_images = background.calc_background(vid_gen, start_frame=start_frame, interval=framerate)
#
#     video_writer = cv.VideoWriter(os.path.join(bg_video_dir, filename), cv.VideoWriter_fourcc(*"mp4v"), 1,
#                                   vid_reader.img_shape[:2][::-1])
#
#     for img, frame in bg_images:
#         video_writer.write(img.astype(np.uint8))
#
#     video_writer.release()

fixed_bg_video_dir = "/data/aicity/train_negatives_f"

for filename in sorted(os.listdir(bg_video_dir)):
    vid_id = int(filename[:-4])

    if vid_id not in negative_video_ids:
        continue

    print("Video:", vid_id)
    # raw
    vid_path = os.path.join(raw_video_dir, filename)
    vid_reader = utils.VideoReaderQueue(vid_path, queue_size=4)
    vid_gen, frames = vid_reader.load_video()
    bg_images = background.calc_background(vid_gen, start_frame=1, interval=framerate)

    # old bg
    bg_vid_path = os.path.join(bg_video_dir, filename)
    bg_vid_reader = utils.VideoReaderQueue(bg_vid_path, queue_size=4)
    bg_vid_gen, frames = bg_vid_reader.load_video()
    # fixed_bg_images = background.calc_background(bg_vid_gen, start_frame=start_frame, interval=framerate)

    video_writer = cv.VideoWriter(os.path.join(fixed_bg_video_dir, filename), cv.VideoWriter_fourcc(*"mp4v"), 1,
                                  bg_vid_reader.img_shape[:2][::-1])
    for img, frame in bg_images:
        print("a", frame)
        if frame < 500:  # dont write 1st 1000 frames
            continue
        elif frame < 1000 + 15 * 30:
            video_writer.write(img.astype(np.uint8))
        else:
            break

    for img, frame in zip(*bg_vid_reader.load_video()):
        if frame < 15:
            continue
        print("b", frame)
        video_writer.write(img)

    video_writer.release()
