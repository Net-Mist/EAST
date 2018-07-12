import time
import numpy as np
import locality_aware_nms as nms_locality
import lanms
import model
from icdar import restore_rectangle
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import cv2
import tftools


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(
            max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(
        xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    # print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape(
            (-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def preprocess_image(image_path, max_side_len=900):
    '''
    Load and pre-process the image.
    If one side is bigger than max_side_len, resize the image.
    ----
    Inputs:
    image_path = Path to the image, str
    max_side_len = Maximum size length of the image to avoid lag, int
    ----
    Outputs:
    image_ori = Original image loaded in an array, numpy.array
    image_resized = Image to be analysed for box detection, numpy.array
    '''
    image_ori = plt.imread(image_path)
    image_resized, (ratio_h, ratio_w) = resize_image(
        image_ori, max_side_len=max_side_len)

    return image_ori, image_resized


def load_frozen_model(frozen_path):
    ''''
    Load the EAST model with its session and variables.
    ----
    Inputs:
    frozen_path = Path to the saved frozen model, str
    ----
    Outputs:
    sess = Tensorflow Session
    input_images = Images placeholder
    fscore = EAST score map placeholder
    fgeometry = EAST geometry placeholder
    '''
    tf.reset_default_graph()
    tftools.model_loader.load_frozen_graph(frozen_path, 'east')
    graph = tf.get_default_graph()
    input_images = graph.get_tensor_by_name('east/input_images:0')
    fscore = graph.get_tensor_by_name('east/feature_fusion/Conv_7/Sigmoid:0')
    fgeometry = graph.get_tensor_by_name('east/feature_fusion/concat_3:0')
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    return sess, input_images, fscore, fgeometry


def load_model(checkpoint_path):
    ''''
    Load the EAST model with its session and variables.
    ----
    Inputs:
    checkpoint_path = Path to the saved model dir, str
    ----
    Outputs:
    sess = Tensorflow Session
    input_images = Images placeholder
    fscore = EAST score map placeholder
    fgeometry = EAST geometry placeholder
    '''
    tf.reset_default_graph()
    # Loading the model
    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        fscore, fgeometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(
            0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
        model_path = os.path.join(checkpoint_path, os.path.basename(
            ckpt_state.model_checkpoint_path))
        print('Restore from {}'.format(model_path))
        saver.restore(sess, model_path)

        return sess, input_images, fscore, fgeometry


def detectboxes(image_resized, east_session,
                image_placeholder, f_score_placeholder, f_geometry_placeholder):
    '''
    Detect the boxes of a given image using the EAST model loaded in a session.
    ----
    Inputs:
    image_path = Path of the image to be analysed, str
    east_session = Session of the east algorithm,tensorflow.python.client.session.Session
    input_images
    ----
    Outputs:
    dictboxes = Dict of (box_id: box), dict
    '''
    # start_time = time.time()
    timer = {'net': 0, 'restore': 0, 'nms': 0}
    start = time.time()
    score, geometry = east_session.run([f_score_placeholder, f_geometry_placeholder], feed_dict={
        image_placeholder: [image_resized]})
    timer['net'] = time.time() - start
    boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
    if boxes is not None:
        dictboxes = {str(i): box[:8].reshape((-1, 4, 2))
                     for i, box in enumerate(boxes)}
    else:
        dictboxes = {}
    # duration = time.time() - start_time

    return dictboxes, score, geometry


def crop_image(image_ori, image_resized, dictboxes, margin):
    '''
    Crops all the boxes in the original image and save them in output_dir.
    We need to know the shape on which the boxes were found to know
    where to crop in the higher res image_ori.
    ----
    Inputs:
    image_ori = Original image before resizing, numpy.array
    image_resized = Image on which detectboxes was run, numpy.array
    dictboxes = Dict of (box_id: box), dict
    margin = Number of pixel to add on each side, int
    ----
    Outputs:
    dictcropims = Dict of crop images {box_id: cropim}, dict
    '''
    ori_im_shape = np.array(image_ori.shape)[:2]
    analyzed_im_shape = np.array(image_resized.shape)[:2]
    # What to multyply the coords of boxes with
    change_coeff = ori_im_shape / analyzed_im_shape
    # Make sure the coeff are in good order
    change_coeff = change_coeff[::-1]
    # Where we store the crops
    dictcropims = {}
    for box_id, bo in dictboxes.items():
        # First apply the change coeff
        box = change_coeff * bo
        # Define coords for croping with a margin.
        # Verify to not exceed the boudaries of the image
        ymin = int(max(np.min(box[:, :, 0]) - margin, 0))
        ymax = int(min(np.max(box[:, :, 0]) + margin, image_ori.shape[1]))
        xmin = int(max(np.min(box[:, :, 1]) - margin, 0))
        xmax = int(min(np.max(box[:, :, 1]) + margin, image_ori.shape[0]))
        crop_im = image_ori[xmin:xmax, ymin:ymax].copy()
        if crop_im.shape[0] > 0. and crop_im.shape[1] > 0.:
            dictcropims[box_id] = crop_im

    return dictcropims


def crop_image_ratio(image_ori, image_resized, dictboxes, margin_ratio=0.1, rotate=False):
    '''
    Crops all the boxes in the original image and save them in output_dir.
    We need to know the shape on which the boxes were found to know
    where to crop in the higher res image_ori.
    ----
    Inputs:
    image_ori = Original image before resizing, numpy.array
    image_resized = Image on which detectboxes was run, numpy.array
    dictboxes = Dict of (box_id: box), dict
    margin_ratio = Percentage of box width/height to add, float
    ----
    Outputs:
    dictcropims = Dict of crop images {box_id: cropim}, dict
    '''
    ori_im_shape = np.array(image_ori.shape)[:2]
    analyzed_im_shape = np.array(image_resized.shape)[:2]
    # What to multyply the coords of boxes with
    change_coeff = ori_im_shape / analyzed_im_shape
    # Make sure the coeff are in good order
    change_coeff = change_coeff[::-1]
    # Where we store the crops
    dictcropims = {}
    for box_id, bo in dictboxes.items():
        # First apply the change coeff
        box = change_coeff * bo
        # Define coords for croping with a margin.
        # Verify to not exceed the boudaries of the image
        width = np.max(box[:, :, 0]) - np.min(box[:, :, 0])
        height = np.max(box[:, :, 1]) - np.min(box[:, :, 1])
        margin_wi = int(margin_ratio * width)
        margin_he = int(margin_ratio * height)
        ymin = int(max(np.min(box[:, :, 0]) - margin_wi, 0))
        ymax = int(min(np.max(box[:, :, 0]) + margin_wi, image_ori.shape[1]))
        xmin = int(max(np.min(box[:, :, 1]) - margin_he, 0))
        xmax = int(min(np.max(box[:, :, 1]) + margin_he, image_ori.shape[0]))
        # Crop the image
        crop_im = image_ori[xmin:xmax, ymin:ymax].copy()
        if crop_im.shape[0] > 0. and crop_im.shape[1] > 0.:
            dictcropims[box_id] = crop_im

    return dictcropims
