import collections
import queue
import threading
import time
import cropping

import cv2 as cv
import numpy as np
import pandas as pd
import os

d = [[-1, 0], [1, 0], [0, 1], [0, -1]]


def search_region(G, pos):
    x1, y1, x2, y2 = pos[1], pos[0], pos[1], pos[0]
    Q = set()
    Q.add(pos)
    h, w = G.shape
    visited = np.zeros((h, w))
    visited[pos] = 1
    while Q:
        u = Q.pop()
        for move in d:
            row = u[0] + move[0]
            col = u[1] + move[1]
            if 0 <= row < h and 0 <= col < w and G[row, col] == 1 and visited[row, col] == 0:
                visited[row, col] = 1
                Q.add((row, col))
                x1 = min(x1, col)
                x2 = max(x2, col)
                y1 = min(y1, row)
                y2 = max(y2, row)
    return [int(x1), int(y1), int(x2), int(y2)]


def compute_iou(region, dets):
    if dets.shape[0] == 0:
        return 0

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order = scores.argsort()[::-1]
    # scores = scores[order]

    xx1 = np.maximum(region[0], x1)
    yy1 = np.maximum(region[1], y1)
    xx2 = np.minimum(region[2], x2)
    yy2 = np.minimum(region[3], y2)

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (areas + (region[2] - region[0] + 1) * (region[3] - region[1] + 1) - inter)
    max_iou = np.max(ovr)
    return max_iou


def compute_brightness(im):
    brightness = np.mean(0.299 * im[:, :, 2] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 0]) / 255.0
    return brightness


def anomaly_nms(all_results, iou_thresh=0.8):
    """
    Applies Non-maximal Supression to a list of anomalies. Resolves duplicate anomalies.

    all_results: list of anomalies [{"region": [x1, y1, x2, y2], "score": _, "start_time": _, "end_time": _}, ...]
    iou_thresh: intersection over union threshold to consider anomalies the same.

    """

    anomalies = np.array([[*res["region"], res["score"], res["start_time"], res["end_time"]]
                          for res in all_results])

    x1 = anomalies[:, 0]
    y1 = anomalies[:, 1]
    x2 = anomalies[:, 2]
    y2 = anomalies[:, 3]
    scores = anomalies[:, 4]
    start_time = anomalies[:, 5]
    end_time = anomalies[:, 6]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]  # sort by score
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])  # compute IoU
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / union

        inds = np.where(iou > iou_thresh)[0]  # select overlapping boxes
        tmp_order = order[inds + 1]
        if len(tmp_order) > 0:
            anomalies[i, 5] = np.min(start_time[tmp_order])  # take the widest time window
            anomalies[i, 6] = np.max(end_time[tmp_order])

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    anomalies = anomalies[keep]
    return anomalies


def get_overlapping_time(anomaly_results, gap_threshold=1):
    """
    Turns the overlapping anomaly detection results into anomaly event times.

    eg. for anomalies starting/ending at times: (0, 100), (50, 200), (300, 400), (390, 800)
        it will return: (0, 200), (300, 800)


    anomaly_results: DataFrame with columns: "start_time", "end_time"
    gap_threshold: events with gaps less than this are merged into one.

    """

    starts = anomaly_results["start_time"].values
    ends = anomaly_results["end_time"].values

    # Sort by (start, end)
    order = ends.argsort()  # sort by end time
    starts = starts[order]
    ends = ends[order]
    order = starts.argsort()  # sort by start time
    starts = starts[order]
    ends = ends[order]

    events = []
    i = 0
    while i < len(starts):
        event_start = starts[i]
        event_end = ends[i]
        while i + 1 < len(starts) and starts[i + 1] - event_end < gap_threshold:  # find the end of the overlap
            if ends[i + 1] > event_end:  # take later end point
                event_end = ends[i + 1]

            i += 1

        events.append((event_start, event_end))
        i += 1

    return events



def mask(arr, msk):
    """
    Just makes it clear where masking is happening, so it is not confused with normal multiplication

    arr: np.array of data
    mask: boolean np.array with same shape as arr
    """

    return arr * msk


class ResultsDict(collections.OrderedDict):
    """
    Accumulates detection results as they are generated.

    results_gen: Generator that yields (key, results) pairs
    args, kwargs: extra arguments to pass to the dictionary

    """


    def __init__(self, results_gen, results_getter=None, name="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_gen = results_gen
        self.max_frame = -1
        self.name = name
        self.results_getter = results_getter


    def __getitem__(self, key):
        """
        Retrieve the results for given key.
        If the key is not in the dictionary, generate results up to the key (frame).
        Returns None if the frame is not produced from the generator
        """
        assert type(key) == int  # only support integer frame number for now

        if key in self:
            return super().__getitem__(key)

        else:
            # Generate new results
            # todo: remove the need to generate all preceeding results if there is a large gap between calls
            while self.max_frame < key:
                try:
                    frame, results = next(self.results_gen)
                except StopIteration:
                    print(f"{self.name}, max {self.max_frame}, key: {key}")
                    break

                print(f"generated {self.name}: {frame}")
                self[frame] = results
            else:
                if key in self:
                    return self[key]

            # Ask for exact result
            if key not in self and self.results_getter is not None:
                frame, results = self.results_getter(key)
                print(f"generated {self.name}: {frame}")
                self[frame] = results

            # Last resort
            if key not in self:
                print(self.name, "Key not found:", key)
                return None
                # raise KeyError


    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.max_frame = max(key, self.max_frame)


    def iterator(self):
        """
        Returns an iterator over all items, generates new results too.
        """

        for key, value in self.items():
            yield key, value

        for frame, results in self.results_gen:
            print(f"generated {self.name}: {frame}")
            self[frame] = results
            yield frame, results

    def gen_next(self):
        """
        Forces the generation of the next result
        :return:
        """

        try:
            frame, results = next(self.results_gen)
            self[frame] = results
            print(f"generated {self.name}: {frame}")
            return frame, results
        except StopIteration:
            print(f"Could not generate {self.name}, generator depleted.")



    @staticmethod
    def from_df(df, key_col="frame"):
        return ResultsDict(df.groupby(key_col))





class VideoReader:
    def __init__(self, filename):
        self.filename = filename

        self.nframes = None
        self.framerate = None
        self.img_shape = None
        self._set_video_info()


    def load_video(self, start_frame=0, end_frame=None, interval=1):
        """
        Loads the images of the video
        Returns a generator with the images, and the corresponding frame numbers.

        interval: Interval between frames returned. eg. 1 = every frame, 20 = every 20th frame.
        """


        def read_frames():
            vid = cv.VideoCapture(self.filename)

            for i in frame_nums:
                vid.set(cv.CAP_PROP_POS_FRAMES, i)

                has_frame, img = vid.read()

                if has_frame:
                    yield img
                else:
                    break

            vid.release()


        frame_nums = range(start_frame, end_frame or self.nframes, interval)

        return read_frames(), frame_nums


    def get_frame(self, n):
        """
        Returns the image at a specific frame number
        """

        vid = cv.VideoCapture(self.filename)
        vid.set(cv.CAP_PROP_POS_FRAMES, n)

        _, img = vid.read()

        vid.release()
        return img


    def _set_video_info(self):
        vid = cv.VideoCapture(self.filename)
        _, img = vid.read()

        self.nframes = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
        self.framerate = vid.get(cv.CAP_PROP_FPS)
        self.img_shape = img.shape

        vid.release()


class VideoReaderQueue(VideoReader):
    def __init__(self, filename, queue_size=32):
        super().__init__(filename)
        self.queue_size = queue_size


    def load_video(self, interval=1):
        """
        Loads the images of the video
        Returns a generator with the images, and the corresponding frame numbers.
        Loads the video on a separate thread lazily.

        interval: Interval between frames returned. eg. 1 = every frame, 20 = every 20th frame.
        """


        def read_frames():
            vid = cv.VideoCapture(self.filename)

            i = 0
            while vid.isOpened():
                grabbed = vid.grab()

                if grabbed:
                    if i in frame_nums:
                        _, img = vid.retrieve()
                        q.put(img)
                else:
                    break

                i += 1

            q.put(None)  # Mark end of video
            vid.release()


        def yield_frames():
            while True:
                if q.qsize() > 0:
                    item = q.get()

                    if item is None:
                        break
                    else:
                        yield item
                else:
                    time.sleep(0.01)


        frame_nums = range(0, self.nframes, interval)

        q = queue.Queue(maxsize=self.queue_size)
        thread = threading.Thread(target=read_frames)
        thread.daemon = True
        thread.start()

        return yield_frames(), frame_nums


class ImageReader:
    def __init__(self, folder):
        self.folder = folder

        self.filenames = sorted(os.listdir(folder),
                                key=lambda f: int(f[:-4]))  # "123.jpg" -> sort by 123 instead of the full string


    def load_images(self):
        """
        Loads the images, returning them as a generator.

        """
        for filename in self.filenames:
            file_path = os.path.join(self.folder, filename)
            img = cv.imread(file_path)

            yield img
