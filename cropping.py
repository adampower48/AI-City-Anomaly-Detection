import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model


def do_regression(df, plot=False, max_area=250, outlier_std=2):
    """
    Does linear regression of y-position vs vehicle height
    h = a * y + b

    df: DataFrame with columns: [x1, y1, x2, y2], the positions of the bounding boxes
    plot: plot the regression results
    max_area: dont consider any regions bigger than this
    outlier_std: standard deviations above/below regression line.
                Values outside this range are removed and regression is re-calculated.
    """

    y = (df["y1"] + df["y2"]) / 2  # center y coord of bbox
    h = np.sqrt((df["y2"] - df["y1"]) * (df["x2"] - df["x1"]))  # sqrt bbox area

    keep = np.isfinite(h) & (h < max_area)

    colors = df.loc[keep, "score"].values
    y = y[keep]  # Remove some wonky values
    h = h[keep]

    regression = linear_model.LinearRegression()

    regression.fit(
        np.array(y).reshape(-1, 1),  # (n, 1)
        np.array(h)  # (n)
    )

    # remove outliers and refit
    pred = regression.predict(np.array(y).reshape(-1, 1))
    diff = (h - pred).values

    # remove outliers
    keep = np.abs((diff - diff.mean()) / diff.std()) < outlier_std  # keep data within 2 standard deviations
    colors = colors[keep]
    y = y[keep]
    h = h[keep]

    regression.fit(
        np.array(y).reshape(-1, 1),  # (n, 1)
        np.array(h)  # (n)
    )

    a, b = regression.coef_[0], regression.intercept_

    if plot:
        f = lambda x: a * x + b
        # plot points
        order = colors.argsort()
        y = y.iloc[order]
        h = h.iloc[order]
        colors = colors[order]
        plt.scatter(y, h, c=colors)
        plt.plot([0, max(y)], [f(0), f(max(y))])
        plt.xlabel("y position")
        plt.ylabel("sqrt bounding box area")
        plt.show()

        print(f"a: {a}, b: {b}")

    return a, b


def generate_box_row(y_min, box_height, box_width, img_width, exclude_last=False, clip_last=True, min_width=10):
    """
    Creates bounding boxes for a single row

    y_min: base y-coordinate for boxes
    box_height:
    box_width:
    img_width:
    exclude_last: whether or not to create the last box of the row, which usualy overlaps outside the image boundary.
    clip_last: if true, chops the end off the last box of the row. if exclude_last == false, this has no effect.
    min_width: Does not generate boxes less than this width.

    """

    num_boxes = int(np.ceil(img_width / box_width))
    x_positions = np.linspace(0, box_width * num_boxes, num_boxes + 1)

    if exclude_last:  # exclude box that would be partially outside image
        x_positions = x_positions[:-1]

    boxes = []
    for x_min, x_max in zip(x_positions[:-1], x_positions[1:]):
        if clip_last and x_max >= img_width:  # clip last box to image boundary
            x_max = img_width - 1

        if x_max - x_min < min_width:  # skip boxes that would be too small
            continue

        boxes.append([int(x_min), y_min, int(x_max), y_min + box_height])

    return boxes


def generate_crop_boxes(min_height, a_reg, b_reg, img_shape, row_capacity=3, box_aspect_ratio=2, exclude_last=False,
                        clip_last=True, min_width=10):
    """
    Creates all crop boxes for the image.

    min_height: minimum vehicle height
    a_reg: linear regression coefficient for y-pos vs height
    b_reg: linear regression intercept for y-pos vs height
    img_shape: (height, width) of image
    row_capacity: Vehicle capacity for each row. Not 100% sure how to explain this, see paper.
    box_aspect_rato: width/height bounding box ratio
    exclude_last, clip_last, min_width: see generate_box_row function.

    """


    def exp_func(x):
        # Modified function, seems to work better/properly
        # capacity space -> height space
        #         return int(np.exp(x) / a_reg) # theirs
        return np.exp(a_reg * x)  # mine


    def log_func(x):
        # height space -> capacity space
        return np.log(x) / a_reg


    def f(y):
        # h = a * y + b
        # y position -> vehicle height at that position
        return a_reg * y + b_reg


    def f_inv(h):
        # vehicle height -> y position
        return int((h - b_reg) / a_reg)


    # k * ln(k*y2+b) - k * ln(k*y1+b) this corresponds to the big integral in the paper
    min_height = max(min_height, f(0))
    total_capacity = log_func(f(img_shape[0])) - log_func(min_height)

    ###
    # heights = list(map(f, range(0, img_shape[0])))
    # caps = list(map(log_func, heights))
    # plt.plot(caps)
    # plt.show()
    ###

    num_rows = int(np.ceil(total_capacity / row_capacity))
    stride_cap = total_capacity / num_rows  # capacity stride

    start_capacity = log_func(min_height)
    print(total_capacity, num_rows, stride_cap, start_capacity)
    vert_capacities = np.linspace(start_capacity, start_capacity + stride_cap * num_rows, num_rows + 1)
    print("v", vert_capacities)

    # convert to y coord
    #     y_positions = list(map((lambda x: exp_func(a_reg * x - b_reg)), vert_capacities)) # theirs
    y_heights = list(map(exp_func, vert_capacities))  # mine
    print("yh", y_heights)
    y_positions = list(map(f_inv, y_heights))
    y_positions = [y for y in y_positions if y > 0]
    print("yp", y_positions)

    boxes = []
    for y_min, y_max in zip(y_positions[:-1], y_positions[1:]):
        box_width = (y_max - y_min) * box_aspect_ratio

        boxes += generate_box_row(y_min, y_max - y_min, box_width, img_shape[1], exclude_last, clip_last, min_width)

    return boxes


def crop_image(img, crop_boxes):
    """
    Chops up an image into boxes

    img: PIL Image
    crop_boxes: list of boxes [x1, y1, x2, y2]
    """

    return [img.crop(box) for box in crop_boxes]


def resize_crops(crops, threshold=0.01):
    """
    Resizes images to the biggest in the list.
    Maintains aspect ratio, pads if necessary.

    crops: list of cropped PIL images
    threshold: determines if an image has been chopped off
    """

    # Uses area to determine biggest image, might not work well if the biggest image is a weird shape off a bit
    biggest_size = max((img.size for img in crops), key=np.prod)
    biggest_aspect = biggest_size[0] / biggest_size[1]

    resized = []
    for img in crops:
        aspect = img.size[0] / img.size[1]

        # chopped off image -> scale and pad
        if abs(biggest_aspect - aspect) > threshold:
            scaled = img.resize((int(biggest_size[0] * aspect / biggest_aspect), biggest_size[1]))

            new = PIL.Image.new("RGB", biggest_size, (0, 0, 0))
            new.paste(scaled, scaled.getbbox())

        # normal image -> scale
        else:
            new = img.resize(biggest_size)

        resized.append(new)

    return resized, biggest_size


def create_crop_boxes(results_path, crop_boxes_path, img_shape, min_object_size=10, row_capacity=3,
                      crop_box_aspect_ratio=2):
    """
    Creates crop boxes based off of detection results.


    :param results_path: path to detection results csv
    :param crop_boxes_path: path to save crop boxes
    :param img_shape: (h, w) of image
    :param min_object_size, row_capacity, crop_box_aspect_ratio: see generate_crop_boxes function
    """


    # Read bboxes
    bbox_df = pd.read_csv(results_path)
    a, b = do_regression(df=bbox_df, plot=False)

    # Create crop boxes
    crop_boxes = generate_crop_boxes(min_object_size, a, b, img_shape, row_capacity, crop_box_aspect_ratio)

    # Save crop boxes
    pd.DataFrame(crop_boxes, columns=["x1", "y1", "x2", "y2"]).to_csv(crop_boxes_path, index=False)


def crop_box_generator(results_dict, img_shape, min_object_size=10, row_capacity=3, crop_box_aspect_ratio=1.67,
                       score_thresh=0.1, start_frame=100, max_area=900000, update_interval=5, reg_outlier_std=3,
                       max_points=250, min_points=10):
    """
    Creates updated crop boxes from detection results.

    :param results_dict: ResultsDict object with results
    :param img_shape: image shape
    :param min_object_size, row_capacity, crop_box_aspect_ratio: see generate_crop_boxes function
    :param score_thresh: Threshold for choosing what results to use for crop boxes
    :param start_frame: Start generating crop boxes after this many frames
    :param max_area: Max sqrt bbox area to use for regression
    :param min_points: Minimum valid points needed to generate boxes
    :param max_points: Max number of points to use for regression, takes best score points first
    :param reg_outlier_std: see do_regression function
    :param update_interval: interval between crop box updates

    :return: generator that yields the most up to date crop boxes as a list
    """

    last_boxes = []
    i = 0
    while True:
        i += 1

        # skip if before start frame
        if results_dict.max_frame < start_frame:
            yield []
            continue

        # skip if its not time to update yet
        if i % update_interval != 0:
            yield last_boxes
            continue

        # Generate new boxes
        try:
            # todo: this could get pretty memory/performance intensive later.
            #       Find a way to update regression results instead of generating from scratch each time.
            merged_results = pd.concat(results_dict.values())
            print(len(merged_results))
            merged_results = merged_results[merged_results["score"] > score_thresh
                                            ].sort_values("score", ascending=False).iloc[:max_points, :]
            print(merged_results)

            if len(merged_results) > min_points:
                a_reg, b_reg = do_regression(merged_results, plot=(i % update_interval == 0), max_area=max_area,
                                             outlier_std=reg_outlier_std)
                print("a:", a_reg, "b:", b_reg)
                last_boxes = generate_crop_boxes(min_object_size, a_reg, b_reg, img_shape, row_capacity,
                                                 crop_box_aspect_ratio)
            yield last_boxes

        except FileNotFoundError as e:  # Exception as e:
            print(type(e), e)
            yield []  # full image


def cropped_detection_to_original(bbox, crop_box, resized_shape):
    """
    Takes bounding boxes from detection on cropped, resized images.
    Translates them to their position on the original image
    Assumes crop boxes were not chopped off at the edge of the picture. todo


    bbox: detected bounding box on image after cropping/resizing
    crop_box: bounding box generated by generate_crop_boxes function
    resized_shape: (height, width) of image after resizing.

    """

    # Turn into numpy arrays for easier computation
    bbox = np.reshape(bbox, (2, 2))
    crop_box = np.reshape(crop_box, (2, 2))
    resized_shape = np.array(resized_shape)

    # Calculate scales
    crop_shape = crop_box[1] - crop_box[0]
    resize_scale = resized_shape / crop_shape

    # Translate bounding box
    bbox_original = bbox / resize_scale + crop_box[0]

    return bbox_original.reshape((4,))
