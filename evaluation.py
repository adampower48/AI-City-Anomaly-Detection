"""
Evaluation with AI City metrics. They only described their evaluation code, didnt release it,
so this may not be accurate.
"""


import os
import pandas as pd

import utils
import numpy as np


def get_event_iou(anomaly_times, gt_times):
    """
    Get intersection over union for detection results and ground truth.
    todo: check if this is actually correct

    :param anomaly_times: [(start1, end1), (start2, end2), ...]
    :param gt_times: [(start1, end1), (start2, end2), ...]
    :return: float [0, 1]
    """


    def get_intersections():
        _intersections = []

        i = 0
        j = 0
        while i < len(anomaly_times) and j < len(gt_times):
            cur_start = max(anomaly_times[i][0], gt_times[j][0])
            cur_end = min(anomaly_times[i][1], gt_times[j][1])

            _intersections.append((cur_start, cur_end))

            if cur_end == anomaly_times[i][1]:
                i += 1
            if cur_end == gt_times[j][1]:
                j += 1

        return _intersections


    def get_unions():
        combined = sorted(anomaly_times + gt_times)
        _unions = []

        i = 0
        while i < len(combined):
            event_start = combined[i][0]
            event_end = combined[i][1]
            while i + 1 < len(combined) and combined[i + 1][0] < event_end:  # find the end of the overlap
                if combined[i + 1][1] > event_end:  # take later end point
                    event_end = combined[i + 1][1]

                i += 1

            _unions.append((event_start, event_end))
            i += 1

        return _unions


    anomaly_times.sort()
    gt_times.sort()

    intersections = get_intersections()
    unions = get_unions()

    intersection = sum(end - start for start, end in intersections)
    union = sum(end - start for start, end in unions)

    if union == 0:
        return 0

    return intersection / union


def get_results_matrix(anomaly_times, gt_times, length=892):
    """
    Gets True/False Positives/Negatives

    :param length: video length in seconds
    :param anomaly_times: [(start1, end1), (start2, end2), ...]
    :param gt_times: [(start1, end1), (start2, end2), ...]
    :return: TP, FP, TN, FN
    """

    anomaly_times.sort()
    gt_times.sort()

    anomaly_times = np.array(anomaly_times).flatten().tolist()
    gt_times = np.array(gt_times).flatten().tolist()

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    cur_an = False
    cur_gt = False

    i = 0
    j = 0
    prev_time = 0
    cur_time = 0
    while i < len(anomaly_times) and j < len(gt_times):
        change_an = False
        change_gt = False

        if anomaly_times[i] < gt_times[j]:
            change_an = True
            cur_time = anomaly_times[i]
            i += 1
        elif anomaly_times[i] > gt_times[j]:
            change_gt = True
            cur_time = gt_times[j]
            j += 1
        else:
            change_an = True
            change_gt = True
            cur_time = anomaly_times[i]
            i += 1
            j += 1

        if cur_an:
            if cur_gt:
                tp += cur_time - prev_time
            else:
                fp += cur_time - prev_time
        else:
            if cur_gt:
                fn += cur_time - prev_time
            else:
                tn += cur_time - prev_time

        if change_an:
            cur_an = not cur_an
        if change_gt:
            cur_gt = not cur_gt

        prev_time = cur_time

    # Finish the other list
    while i < len(anomaly_times):
        cur_time = anomaly_times[i]
        i += 1

        if cur_an:
            if cur_gt:
                tp += cur_time - prev_time
            else:
                fp += cur_time - prev_time
        else:
            if cur_gt:
                fn += cur_time - prev_time
            else:
                tn += cur_time - prev_time

        cur_an = not cur_an
        prev_time = cur_time

    while j < len(gt_times):
        cur_time = gt_times[j]
        j += 1

        if cur_an:
            if cur_gt:
                tp += cur_time - prev_time
            else:
                fp += cur_time - prev_time
        else:
            if cur_gt:
                fn += cur_time - prev_time
            else:
                tn += cur_time - prev_time

        cur_gt = not cur_gt
        prev_time = cur_time

    # Finish to the end of the video
    if cur_time < length:
        cur_time = length
        if cur_an:
            if cur_gt:
                tp += cur_time - prev_time
            else:
                fp += cur_time - prev_time
        else:
            if cur_gt:
                fn += cur_time - prev_time
            else:
                tn += cur_time - prev_time

    return tp, fp, tn, fn


def get_results_matrix_within(anomaly_times, gt_times, length=892, max_gap=10):
    anomaly_times = sorted(anomaly_times)  # sort and copy
    gt_times.sort()

    # anomaly_times = np.array(anomaly_times).flatten().tolist()
    # gt_times = np.array(gt_times).flatten().tolist()

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    errors = []

    for gt_start, gt_end in gt_times:
        i = min(range(len(anomaly_times)), key=lambda j: abs(gt_start - anomaly_times[j][0]), default=None)
        if i is None:
            fn += 1
            continue

        an_start, an_end = anomaly_times.pop(i)

        if abs(gt_start - an_start) <= max_gap:  # within 10 seconds
            tp += 1
            errors.append(gt_start - an_start)
        else:
            fp += 1

    fp += len(anomaly_times)

    return tp, fp, tn, fn, errors


if __name__ == '__main__':

    anomaly_results_dir = "/data/aicity/winner_team/anomaly_results/train"
    ground_truth_results_path = "/data/aicity/train-anomaly-results.txt"

    # Read & parse ground truth
    gt_results = pd.read_csv(ground_truth_results_path, header=None, names=["video_id", "start", "end"], sep=" ")
    gt_results["event"] = gt_results[["start", "end"]].apply((lambda x: [tuple(x)]), axis=1)
    gt_results = gt_results.groupby("video_id")["event"].sum()
    # print(gt_results)

    precisions = []
    recalls = []
    for i in range(100):
        score_thresh = i / 100
        tps, fps, tns, fns = 0, 0, 0, 0
        errors = []
        for video_id in range(1, 101, 1):
            filename = f"{video_id}.csv"
            # for filename in sorted(os.listdir(anomaly_results_dir)):
            #     video_id = int(filename[:-4])

            try:
                path = os.path.join(anomaly_results_dir, filename)
                an_results = pd.read_csv(path)
                an_results = an_results[an_results.score > score_thresh]

                # anomaly_events = utils.get_overlapping_time(an_results)
                # print(video_id, anomaly_events)
                anomaly_events = list(zip(an_results["start_time"].values, an_results["end_time"].values))
            except FileNotFoundError:
                anomaly_events = []

            try:
                gt_events = gt_results[video_id]
            except KeyError:
                gt_events = []

            # tp, fp, tn, fn = get_results_matrix(anomaly_events, gt_events)
            tp, fp, tn, fn, err = get_results_matrix_within(anomaly_events, gt_events)
            # print(video_id, anomaly_events, gt_events, tp, fp, tn, fn, err)

            tps += tp
            fps += fp
            tns += tn
            fns += fn
            errors += err

        print(tps, fps, tns, fns, errors)
        rmse = (np.array(errors) ** 2).mean() ** 0.5
        nrmse = min(rmse / 300, 1)
        recall = tps / (tps + fns)

        if tps + fps > 0:
            precision = tps / (tps + fps)
        else:
            precision = 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        s3_score = f1 * (1 - nrmse)

        precisions.append(precision)
        recalls.append(recall)

        print("score thresh:", score_thresh, "P:", precision, "R:", recall, "F1:", f1, "RMSE:", rmse, "S3:", s3_score)

    print(precisions, recalls)
