import os
import time
import datetime
import cv2
import uuid
import json
import functools
import logging
import collections
import fire
import io

import numpy as np
import tensorflow as tf

from flask import Flask, request, render_template
from evaluating import resize_image, sort_poly, detect

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

app = Flask(__name__)

save_dir = None
session = None
image_placeholder = None
score_output = None
map_output = None


@functools.lru_cache(maxsize=1)
def get_host_info():
    ret = {}
    with open('/proc/cpuinfo') as f:
        ret['cpuinfo'] = f.read()

    with open('/proc/meminfo') as f:
        ret['meminfo'] = f.read()

    with open('/proc/loadavg') as f:
        ret['loadavg'] = f.read()

    return ret


def predictor(img):
    global session
    """
    Returns:
        {
        'text_lines': [
            {'score': , 'x0': , 'y0': , 'x1': , ... 'y3': }
            ....
        ],
        'rtparams': {  # runtime parameters
            'image_size': ,
            'working_size': ,
        },
        'timing': {
            'net': ,
            'restore': ,
            'nms': ,
            'cpuinfo': ,
            'meminfo': ,
            'uptime': ,
        }
    }
    """
    start_time = time.time()
    rtparams = collections.OrderedDict()
    rtparams['start_time'] = datetime.datetime.now().isoformat()
    rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
    timer = collections.OrderedDict([
        ('net', 0),
        ('restore', 0),
        ('nms', 0)
    ])

    # Prepare data
    im_resized, (ratio_h, ratio_w) = resize_image(img)
    rtparams['working_size'] = '{}x{}'.format(im_resized.shape[1], im_resized.shape[0])

    # Network
    start = time.time()
    score, geometry = session.run([score_output, map_output], feed_dict={image_placeholder: [im_resized[:, :, ::-1]]})
    timer['net'] = time.time() - start

    # NMS
    boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
    logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
        timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))

    if boxes is not None:
        scores = boxes[:, 8].reshape(-1)
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    duration = time.time() - start_time
    timer['overall'] = duration
    logger.info('[timing] {}'.format(duration))

    text_lines = []
    if boxes is not None:
        text_lines = []
        for box, score in zip(boxes, scores):
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            tl = collections.OrderedDict(zip(
                ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                map(float, box.flatten())))
            tl['score'] = float(score)
            text_lines.append(tl)
    ret = {
        'text_lines': text_lines,
        'rtparams': rtparams,
        'timing': timer,
    }
    ret.update(get_host_info())
    return ret


@app.route('/')
def index():
    return render_template('index.html', session_id='dummy_session_id')


def draw_illu(illu, rst):
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
    return illu


def save_result(img, rst):
    global save_dir
    session_id = str(uuid.uuid1())
    dirpath = os.path.join(save_dir, session_id)
    os.makedirs(dirpath)

    # save input image
    output_path = os.path.join(dirpath, 'input.png')
    cv2.imwrite(output_path, img)

    # save illustration
    output_path = os.path.join(dirpath, 'output.png')
    cv2.imwrite(output_path, draw_illu(img.copy(), rst))

    # save json data
    output_path = os.path.join(dirpath, 'result.json')
    with open(output_path, 'w') as f:
        json.dump(rst, f)

    rst['session_id'] = session_id
    return rst


@app.route('/', methods=['POST'])
def index_post():
    bio = io.BytesIO()
    request.files['image'].save(bio)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    rst = predictor(img)

    save_result(img, rst)
    return render_template('index.html', session_id=rst['session_id'])


def main(frozen_model_path, port=8769, debug=False, log_dir="static/results"):
    global save_dir, session, image_placeholder, score_output, map_output
    save_dir = log_dir

    if not os.path.exists(frozen_model_path):
        raise RuntimeError('Frozen model `{}` not found'.format(frozen_model_path))

    logger.info('loading model')
    with tf.gfile.GFile(frozen_model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="east")
    session = tf.Session()

    image_placeholder = tf.get_default_graph().get_tensor_by_name("east/input_images:0")
    score_output = tf.get_default_graph().get_tensor_by_name("east/feature_fusion/score_map/Sigmoid:0")
    map_output = tf.get_default_graph().get_tensor_by_name("east/feature_fusion/concat_3:0")

    app.debug = debug
    app.run('0.0.0.0', port)


if __name__ == '__main__':
    fire.Fire(main)
