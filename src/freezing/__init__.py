import tensorflow as tf
from training import model
import fire
import os


def main(checkpoint_path: str, frozen_model_path: str):
    """
    Create a frozen model from the last checkpoint in directory checkpoint_dir
    Args:
        checkpoint_path: path of the checkpoint to load
        frozen_model_path: path of the frozen file to create

    Returns:

    """
    tf.reset_default_graph()

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    f_score, f_geometry = model.model(input_images, is_training=False)

    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    print('Restore from {}'.format(checkpoint_path))
    saver = tf.train.Saver()
    saver.restore(session, checkpoint_path)

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        session,
        tf.get_default_graph().as_graph_def(),
        ['input_images', 'feature_fusion/score_map/Sigmoid', 'feature_fusion/concat_3']
    )
    with tf.gfile.GFile(frozen_model_path, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
