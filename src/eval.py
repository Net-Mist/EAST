import cv2
import time
import fire
import os

import numpy as np
import tensorflow as tf

from evaluating import resize_image, detect, sort_poly


def get_images(test_data_path):
    """
    find image files in test data path
        Returns: list of files found
    """

    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def main(test_data_path, frozen_model_path, output_dir, no_write_images=False):
    """
    Test a frozen model on a set of images
    Args:
        test_data_path: path to the dir containing the images to test
        frozen_model_path: path of the frozen model to load
        output_dir: path of the dir where writing the output images
        no_write_images: do not write image
    """

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(frozen_model_path):
        raise RuntimeError('Frozen model `{}` not found'.format(frozen_model_path))

    with tf.gfile.GFile(frozen_model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="east")
    session = tf.Session()

    image_placeholder = tf.get_default_graph().get_tensor_by_name("east/input_images:0")
    score_output = tf.get_default_graph().get_tensor_by_name("east/feature_fusion/score_map/Sigmoid:0")
    map_output = tf.get_default_graph().get_tensor_by_name("east/feature_fusion/concat_3:0")

    image_path_list = get_images(test_data_path)
    for image_path in image_path_list:
        image = cv2.imread(image_path)[:, :, ::-1]
        start_time = time.time()
        im_resized, (ratio_h, ratio_w) = resize_image(image)

        timer = {'net': 0, 'restore': 0, 'nms': 0}
        start = time.time()
        score, geometry = session.run([score_output, map_output], feed_dict={image_placeholder: [im_resized]})
        timer['net'] = time.time() - start

        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
        print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            image_path, timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))

        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        print('[timing] {}'.format(duration))

        # save to file
        if boxes is not None:
            res_file = os.path.join(output_dir, '{}.txt'.format(os.path.basename(image_path).split('.')[0]))
            with open(res_file, 'w') as f:
                for box in boxes:
                    # to avoid submitting errors
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        continue
                    f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                        box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                    ))
                    cv2.polylines(image[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                  color=(255, 255, 0), thickness=1)
        if not no_write_images:
            img_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(img_path, image[:, :, ::-1])


if __name__ == '__main__':
    # main('/home/seb/tmp/test_east/in',
    #      '/mnt/nas/tf_experiments/ocr/east/frozen/frozen_east.pb',
    #      '/home/seb/tmp/test_east/out')
    fire.Fire(main)
