import glob
import csv
import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from shapely.geometry import Polygon

from training import dataset

import tensorflow as tf

from data_util import GeneratorEnqueuer

import training.gen_geo_map.gen_geo_map as gen_geo_map

tf.app.flags.DEFINE_string('training_data_path', '/data/ocr/icdar2015/', 'training dataset to use')
tf.app.flags.DEFINE_integer('max_image_large_side', 1280, 'max image size of training')
tf.app.flags.DEFINE_integer('max_text_size', 800, 'if the text in the input image is bigger than this, then we resize '
                                                  'the image according to this')
tf.app.flags.DEFINE_integer('min_text_size', 10, 'if the text size is smaller than this, we ignore it during training')
tf.app.flags.DEFINE_float('min_crop_side_ratio', 0.1, 'when doing random crop from input image, the min length of '
                                                      'min(H, W')
tf.app.flags.DEFINE_string('geometry', 'RBOX', 'which geometry to generate, RBOX or QUAD')

FLAGS = tf.app.flags.FLAGS


def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1 - b2) / (k1 - k2)
        y = k1 * x + b1
    return np.array([x, y], dtype=np.float32)


def line_verticle(line, point):
    # get the verticle line from line across point
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1. / line[0], -1,
                        point[1] - (-1 / line[0] * point[0])]
    return verticle


def rectangle_from_parallelogram(poly):
    """
    fit a rectangle from a parallelogram
    :param poly:
    :return:
    """
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1 - p0, p3 - p0) / (np.linalg.norm(p0 - p1) * np.linalg.norm(p3 - p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p0 and p2
            # p0
            p2p3 = dataset.fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)

            new_p3 = line_cross_point(p2p3, p2p3_verticle)
            # p2
            p0p1 = dataset.fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)

            new_p1 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            p1p2 = dataset.fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)

            new_p1 = line_cross_point(p1p2, p1p2_verticle)
            p0p3 = dataset.fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)

            new_p3 = line_cross_point(p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p1 and p3
            # p1
            p2p3 = dataset.fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)

            new_p2 = line_cross_point(p2p3, p2p3_verticle)
            # p3
            p0p1 = dataset.fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)

            new_p0 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p0p3 = dataset.fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)

            new_p0 = line_cross_point(p0p3, p0p3_verticle)
            p1p2 = dataset.fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)

            new_p2 = line_cross_point(p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)


def sort_rectangle(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # 底边平行于X轴, 那么p0为左上角
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        # 找到最低点右边的点
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right]
        [1]) / (poly[p_lowest][0] - poly[p_lowest_right][0]))
        # assert angle > 0
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle / np.pi * 180 > 45:
            # 这个点为p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi / 2 - angle)
        else:
            # 这个点为p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle


def restore_rectangle_rbox(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array(
            [np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(
            rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array(
            [-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(
            rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p,
                            axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p,
                            axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array(
            [np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(
            rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array(
            [np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(
            rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p,
                            axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p,
                            axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1])


def restore_rectangle(origin, geometry):
    return restore_rectangle_rbox(origin, geometry)


def generate_rbox(im_size, polys, ignore_poly_tags, min_text_size):
    """

    Args:
        im_size: (height, width)
        polys: array of shape N * 4 * 2 : N polygons (most of the time rectangle) defined per 4 point of 2 floats
        ignore_poly_tags: array of bool of shape N. If true then the poly is ignore during training
        min_text_size: if the text size is smaller than this, we ignore it during training

    Returns:

    """
    h, w = im_size
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)
    # mask used during training, to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)

    for poly_id in range(len(polys)):
        poly = polys[poly_id]
        ignore_poly_tag = ignore_poly_tags[poly_id]

        r = [None, None, None, None]
        for i in range(4):
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]), np.linalg.norm(poly[i] - poly[(i - 1) % 4]))

        # score map
        shrinked_poly = dataset.shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, shrinked_poly, 1)
        cv2.fillPoly(poly_mask, shrinked_poly, poly_id + 1)

        # if the poly is too small, then ignore it during training
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        if min(poly_h, poly_w) < min_text_size:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        if ignore_poly_tag:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        xy_in_poly = np.argwhere(poly_mask == (poly_id + 1))

        # Generate a parallelogram for the combination of any two vertices
        fitted_parallelograms = []
        for i in range(4):
            p0 = poly[i]
            p1 = poly[(i + 1) % 4]
            p2 = poly[(i + 2) % 4]
            p3 = poly[(i + 3) % 4]
            edge = dataset.fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            backward_edge = dataset.fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            forward_edge = dataset.fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            if dataset.point_dist_to_line(p0, p1, p2) > dataset.point_dist_to_line(p0, p1, p3):
                # Parallel line p2
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p2[0]]
                else:
                    edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
            else:
                # After p3
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p3[0]]
                else:
                    edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
            # move forward edge
            new_p1 = p1
            new_p2 = line_cross_point(forward_edge, edge_opposite)
            if dataset.point_dist_to_line(p1, new_p2, p0) > dataset.point_dist_to_line(p1, new_p2, p3):
                # across p0
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p0[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
            else:
                # across p3
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p3[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
            new_p0 = line_cross_point(forward_opposite, edge)
            new_p3 = line_cross_point(forward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])

            # or move backward edge
            new_p0 = p0
            new_p3 = line_cross_point(backward_edge, edge_opposite)
            if dataset.point_dist_to_line(p0, p3, p1) > dataset.point_dist_to_line(p0, p3, p2):
                # across p1
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p1[0]]
                else:
                    backward_opposite = [
                        backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
            else:
                # across p2
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p2[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
            new_p1 = line_cross_point(backward_opposite, edge)
            new_p2 = line_cross_point(backward_opposite, edge_opposite)

            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
        areas = [Polygon(t).area for t in fitted_parallelograms]
        parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
        # sort this polygon
        parallelogram_coord_sum = np.sum(parallelogram, axis=1)
        min_coord_idx = np.argmin(parallelogram_coord_sum)
        parallelogram = parallelogram[
            [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

        rectangle = rectangle_from_parallelogram(parallelogram)
        rectangle, rotate_angle = sort_rectangle(rectangle)

        p0_rect, p1_rect, p2_rect, p3_rect = rectangle

        gen_geo_map.gen_geo_map(geo_map, xy_in_poly, rectangle, rotate_angle)

        # for y, x in xy_in_poly:
        #     point = np.array([x, y], dtype=np.float32)
        #     # top
        #     geo_map[y, x, 0] = dataset.point_dist_to_line(p0_rect, p1_rect, point)
        #     # right
        #     geo_map[y, x, 1] = dataset.point_dist_to_line(p1_rect, p2_rect, point)
        #     # down
        #     geo_map[y, x, 2] = dataset.point_dist_to_line(p2_rect, p3_rect, point)
        #     # left
        #     geo_map[y, x, 3] = dataset.point_dist_to_line(p3_rect, p0_rect, point)
        #     # angle
        #     geo_map[y, x, 4] = rotate_angle
    return score_map, geo_map, training_mask


def generator(input_size=512, batch_size=32, background_ratio=3. / 8, random_scale=np.array([0.5, 1, 2.0, 3.0]),
              vis=False):
    image_list = np.array(dataset.get_image_paths(FLAGS.training_data_path))
    print('{} training images in {}'.format(image_list.shape[0], FLAGS.training_data_path))

    image_indexes = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(image_indexes)
        images = []
        image_fns = []
        score_maps = []
        geo_maps = []
        training_masks = []
        for image_index in image_indexes:
            image_path = image_list[image_index]

            # Load the image and the annotation file
            try:
                annotation_file_path = dataset.get_annotation_file_path(image_path)
                image = dataset.load_image(image_path)
                polys, ignore_poly_tags = dataset.load_annotation(annotation_file_path)
            except dataset.ReadingFileError as e:
                print(e)
                continue

            h, w, _ = image.shape
            polys, ignore_poly_tags = dataset.check_and_validate_polys(polys, ignore_poly_tags, (h, w), image_path)

            # random rotate
            # angle = 0 : no rotate
            # ANGLE = 1 : 90
            # ANGLE = 2 : 180
            # ANGLE = 3 : -90
            angle = np.random.randint(0, 4)
            if angle == 1:
                image = np.rot90(image)
                tmp = polys[:, :, 1].copy()
                polys[:, :, 1] = w - polys[:, :, 0].copy()
                polys[:, :, 0] = tmp
            elif angle == 2:
                image = np.rot90(image, k=2)
                tmp = w - polys[:, :, 0].copy()
                polys[:, :, 1] = h - polys[:, :, 1]
                polys[:, :, 0] = tmp
            elif angle == 3:
                image = np.rot90(image, k=3)
                tmp = h - polys[:, :, 1].copy()
                polys[:, :, 1] = polys[:, :, 0]
                polys[:, :, 0] = tmp

            # random scale
            rd_scale = np.random.choice(random_scale)
            image = cv2.resize(image, dsize=None, fx=rd_scale, fy=rd_scale)
            polys *= rd_scale

            # Crop
            crop_backgroud = bool(np.random.rand() < background_ratio)
            image, polys, ignore_poly_tags = dataset.crop_area(image, polys, ignore_poly_tags,
                                                               FLAGS.min_crop_side_ratio,
                                                               crop_background=crop_backgroud)

            # pad the image to the training input size or the longer side of image
            max_h_w_i = np.max([image.shape[0], image.shape[1], input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
            im_padded[:image.shape[0], :image.shape[1], :] = image.copy()
            # Then resize and apply ratio to polygons, if any
            image = cv2.resize(im_padded, dsize=(input_size, input_size))
            polys *= input_size / max_h_w_i

            try:
                score_map, geo_map, training_mask = generate_rbox((input_size, input_size), polys, ignore_poly_tags,
                                                                  FLAGS.min_text_size)
            except Exception as e:
                print(e)
                print(polys)

            try:
                if vis:
                    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
                    # axs[0].imshow(image[:, :, ::-1])
                    # axs[0].set_xticks([])
                    # axs[0].set_yticks([])
                    # for poly in polys:
                    #     poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                    #     poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                    #     axs[0].add_artist(Patches.Polygon(
                    #         poly * 4, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
                    #     axs[0].text(poly[0, 0] * 4, poly[0, 1] * 4, '{:.0f}-{:.0f}'.format(poly_h * 4, poly_w * 4),
                    #                    color='purple')
                    # axs[1].imshow(score_map)
                    # axs[1].set_xticks([])
                    # axs[1].set_yticks([])
                    axs[0, 0].imshow(image[:, :, ::-1])
                    axs[0, 0].set_xticks([])
                    axs[0, 0].set_yticks([])
                    for poly in polys:
                        poly_h = min(
                            abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                        poly_w = min(
                            abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                        axs[0, 0].add_artist(Patches.Polygon(
                            poly, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
                        axs[0, 0].text(
                            poly[0, 0], poly[0, 1], '{:.0f}-{:.0f}'.format(poly_h, poly_w), color='purple')
                    axs[0, 1].imshow(score_map[::, ::])
                    axs[0, 1].set_xticks([])
                    axs[0, 1].set_yticks([])
                    axs[1, 0].imshow(geo_map[::, ::, 0])
                    axs[1, 0].set_xticks([])
                    axs[1, 0].set_yticks([])
                    axs[1, 1].imshow(geo_map[::, ::, 1])
                    axs[1, 1].set_xticks([])
                    axs[1, 1].set_yticks([])
                    axs[2, 0].imshow(geo_map[::, ::, 2])
                    axs[2, 0].set_xticks([])
                    axs[2, 0].set_yticks([])
                    axs[2, 1].imshow(training_mask[::, ::])
                    axs[2, 1].set_xticks([])
                    axs[2, 1].set_yticks([])
                    plt.tight_layout()
                    plt.show()
                    plt.close()

                images.append(image.astype(np.float32))
                image_fns.append(image_path)
                score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                training_masks.append(
                    training_mask[::4, ::4, np.newaxis].astype(np.float32))

                if len(images) == batch_size:
                    yield images, image_fns, score_maps, geo_maps, training_masks
                    images = []
                    image_fns = []
                    score_maps = []
                    geo_maps = []
                    training_masks = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    pass
