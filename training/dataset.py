import os
import glob
import cv2
import csv

import numpy as np


class ReadingFileError(Exception):
    """
    Error when trying to load an image with cv2 and failing
    """
    pass


def get_image_paths(training_data_path):
    """

    Args:
        training_data_path: path of the dir containing the images

    Returns: The list of all the images to load
    """
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(os.path.join(training_data_path, '*.{}'.format(ext))))
    return files


def load_image(image_path):
    """
    Load the image and return it in a RGB format
    Args:
        image_path: Path of the image to load

    Returns: the image

    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        raise ReadingFileError("Error reading image " + image_path)
    return image


def get_annotation_file_path(image_path):
    """
    Compute the annotation file path from the image path
    Args:
        image_path: path of the image to load

    Returns: the pat  to the annotation file
    """
    annotation_file_path = image_path.replace(os.path.basename(image_path).split('.')[1], 'txt')
    if not os.path.exists(annotation_file_path):
        raise ReadingFileError("Annotation file " + annotation_file_path + "does not exist")
    return annotation_file_path


def load_annotation(file_path):
    """
    load annotation from the text file. This file must exist.
    Args:
        file_path: path of the txt file containing the annotations
    Returns:
        polys: array of shape N * 4 * 2 : N polygons (most of the time rectangle) defined per 4 point of 2 floats
        ignore_poly_tag: if the label is '*' of '###' then the human can't read the text and we musn't use it for training
    """
    polys = []
    ignore_poly_tag = []

    with open(file_path, 'r') as annotation_file:
        reader = csv.reader(annotation_file)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                ignore_poly_tag.append(True)
            else:
                ignore_poly_tag.append(False)
    return np.array(polys, dtype=np.float32), np.array(ignore_poly_tag, dtype=np.bool)


def polygon_area(poly):
    """
    compute area of a polygon. See https://www.mathopenref.com/coordpolygonarea.html for explanation
    Args:
        poly: array of shape 4 * 2 : polygon (most of the time rectangle) defined per 4 point of 2 floats
    Returns: The area. If Positive then point are in clockwise order

    """
    edge = [
        (poly[0][0] * poly[1][1]) - (poly[0][1] * poly[1][0]),
        (poly[1][0] * poly[2][1]) - (poly[1][1] * poly[2][0]),
        (poly[2][0] * poly[3][1]) - (poly[2][1] * poly[3][0]),
        (poly[3][0] * poly[0][1]) - (poly[3][1] * poly[0][0])
    ]
    return np.sum(edge) / 2.


def check_and_validate_polys(polys, ignore_poly_tags, shape, image_path):
    """
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    Args:
        polys: array of shape N * 4 * 2 : N polygons (most of the time rectangle) defined per 4 point of 2 floats
        ignore_poly_tags: tag for ignoring the poly during training
        shape: shape (h, w) of the image
        image_path: path of the image, only for debugging and logging purposes
    Returns:
    """

    (h, w) = shape
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, ignore_poly_tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            print('invalid poly :', image_path)
            continue
        if p_area < 0:
            print('poly in wrong direction :', image_path)
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


def crop_area(image, polys, ignore_poly_tags, min_crop_side_ratio, crop_background=False):
    """
    make random crop from the input image
    This function crop the input in a certain way so all the text remain in the output image
    Args:
        image: numpy array of shape H * W * 3
        polys: array of shape N * 4 * 2 : N polygons (most of the time rectangle) defined per 4 point of 2 floats
        ignore_poly_tags: tag for ignoring the poly during training
        min_crop_side_ratio: when doing random crop from input image, the min length of 'min(H, W)'
        crop_background: bool. if True then the result of a cropping can have no text inside. Else it must have some
                        text
    Returns:

    """

    h, w, _ = image.shape

    # The size of the pad is the probability to crop the image from the border
    pad_h = h // 10
    pad_w = w // 10

    # Find position of text and put 1 inside the following table when the text is found
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        x_min = np.min(poly[:, 0])
        x_max = np.max(poly[:, 0])
        y_min = np.min(poly[:, 1])
        y_max = np.max(poly[:, 1])
        w_array[x_min + pad_w:x_max + pad_w] = 1
        h_array[y_min + pad_h:y_max + pad_h] = 1

    # Find all indices without text. Can't be None because of the pad
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]

    # Select the first point of the crop
    x1 = np.clip(np.random.choice(w_axis) - pad_w, 0, w - 1)
    y1 = np.clip(np.random.choice(h_axis) - pad_h, 0, h - 1)

    # Be sure that the second point is not too near of the first one
    x_min = max(0, int(x1 + pad_w - min_crop_side_ratio * w / 2))
    x_max = min(int(x1 + pad_w + min_crop_side_ratio * w / 2), len(w_array))
    w_array[x_min: x_max] = 1
    y_min = max(0, int(y1 + pad_h - min_crop_side_ratio * h / 2))
    y_max = min(int(y1 + pad_h + min_crop_side_ratio * h / 2), len(h_array))
    h_array[y_min:y_max] = 1

    if not crop_background and polys.shape[0] > 0:
        # In this case there must be some text inside the cropped region
        # For that we start by randomly select a poly which will be inside the cropped area and select the last point
        # In function of this poly
        poly_id = np.random.randint(0, len(polys))

        poly = polys[poly_id]

        if x1 < poly[0, 0]:
            # Then the point is on the left of the poly and the second point will need to be on the right
            w_array[: int(pad_w + poly[0, 0])] = 1
        else:
            # Then the point is on the right of the poly and the second point will need to be on the left
            w_array[int(pad_w + poly[0, 0]):] = 1
        if y1 < poly[0, 1]:
            # Then the point on the poly and the second point will need to be bellow
            h_array[: int(pad_h + poly[0, 1])] = 1
        else:
            # Then the point is bellow the poly and the second point will need to be upside
            h_array[int(pad_h + poly[0, 1]):] = 1

    # Find again all indices without text. Can't be None because there are 2 pads in each directions
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]

    # Select the second point of the crop
    x2 = np.clip(np.random.choice(w_axis) - pad_w, 0, w - 1)
    y2 = np.clip(np.random.choice(h_axis) - pad_h, 0, h - 1)

    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)

    if polys.shape[0] != 0:
        # Find all poly that are in the area
        poly_axis_in_area = (polys[:, :, 0] >= x_min) & (polys[:, :, 0] <= x_max) \
                            & (polys[:, :, 1] >= y_min) & (polys[:, :, 1] <= y_max)
        selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
    else:
        selected_polys = []

    polys = polys[selected_polys]
    ignore_poly_tags = ignore_poly_tags[selected_polys]

    polys[:, :, 0] -= x_min
    polys[:, :, 1] -= y_min

    return image[y_min:y_max + 1, x_min:x_max + 1, :], polys, ignore_poly_tags


def shrink_poly(poly, r):
    """
    fit a poly inside the origin poly
    Taken from teh original implementation but I didn't take time to verify this part, maybe bugs here...
    used for generate the score map
    Args:
        poly: the text poly
        r: r in the paper
    :return: the shrinked poly
    """
    # shrink ratio
    shrink_ratio = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
            np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        # p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += shrink_ratio * r[0] * np.cos(theta)
        poly[0][1] += shrink_ratio * r[0] * np.sin(theta)
        poly[1][0] -= shrink_ratio * r[1] * np.cos(theta)
        poly[1][1] -= shrink_ratio * r[1] * np.sin(theta)
        # p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += shrink_ratio * r[3] * np.cos(theta)
        poly[3][1] += shrink_ratio * r[3] * np.sin(theta)
        poly[2][0] -= shrink_ratio * r[2] * np.cos(theta)
        poly[2][1] -= shrink_ratio * r[2] * np.sin(theta)
        # p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]),
                           (poly[3][1] - poly[0][1]))
        poly[0][0] += shrink_ratio * r[0] * np.sin(theta)
        poly[0][1] += shrink_ratio * r[0] * np.cos(theta)
        poly[3][0] -= shrink_ratio * r[3] * np.sin(theta)
        poly[3][1] -= shrink_ratio * r[3] * np.cos(theta)
        # p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]),
                           (poly[2][1] - poly[1][1]))
        poly[1][0] += shrink_ratio * r[1] * np.sin(theta)
        poly[1][1] += shrink_ratio * r[1] * np.cos(theta)
        poly[2][0] -= shrink_ratio * r[2] * np.sin(theta)
        poly[2][1] -= shrink_ratio * r[2] * np.cos(theta)
    else:
        # p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]),
                           (poly[3][1] - poly[0][1]))
        poly[0][0] += shrink_ratio * r[0] * np.sin(theta)
        poly[0][1] += shrink_ratio * r[0] * np.cos(theta)
        poly[3][0] -= shrink_ratio * r[3] * np.sin(theta)
        poly[3][1] -= shrink_ratio * r[3] * np.cos(theta)
        # p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]),
                           (poly[2][1] - poly[1][1]))
        poly[1][0] += shrink_ratio * r[1] * np.sin(theta)
        poly[1][1] += shrink_ratio * r[1] * np.cos(theta)
        poly[2][0] -= shrink_ratio * r[2] * np.sin(theta)
        poly[2][1] -= shrink_ratio * r[2] * np.cos(theta)
        # p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]),
                           (poly[1][0] - poly[0][0]))
        poly[0][0] += shrink_ratio * r[0] * np.cos(theta)
        poly[0][1] += shrink_ratio * r[0] * np.sin(theta)
        poly[1][0] -= shrink_ratio * r[1] * np.cos(theta)
        poly[1][1] -= shrink_ratio * r[1] * np.sin(theta)
        # p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]),
                           (poly[2][0] - poly[3][0]))
        poly[3][0] += shrink_ratio * r[3] * np.cos(theta)
        poly[3][1] += shrink_ratio * r[3] * np.sin(theta)
        poly[2][0] -= shrink_ratio * r[2] * np.cos(theta)
        poly[2][1] -= shrink_ratio * r[2] * np.sin(theta)
    return poly


def fit_line(x, y):
    """
    Compute a line that feet all the point
    TODO it should be much faster to compute (y2 - y1) / (x2 - x1)
    Args:
        x: list of all the x of all the points
        y: list of all the y of all the points

    Returns: (a, b, c) such as the line equation is a*x + b*y + c = 0
    """
    # fit a line ax+by+c = 0
    if x[0] == x[1]:
        return [1., 0., -x[0]]
    else:
        [k, b] = np.polyfit(x, y, deg=1)
        return [k, -1., b]


def point_dist_to_line(p1, p2, p3):
    """
    compute the distance from p3 to p1-p2
    Args:
        p1: point
        p2: point
        p3: point

    Returns: The distance p3, p1p2

    """
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
