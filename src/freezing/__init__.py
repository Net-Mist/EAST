import tensorflow as tf
from training import model


def main(checkpoint_path: str, frozen_model_path: str):
    """
    Create a frozen model from the last checkpoint in directory checkpoint_dir
    Args:
        checkpoint_path: path of the checkpoint to load
        frozen_model_path: path of the frozen file to create

    Returns:

    """
    tf.reset_default_graph()

    input_images = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='input_images')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    print('Restore from {}'.format(checkpoint_path))
    saver.restore(session, checkpoint_path)

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        session,
        tf.get_default_graph().as_graph_def(),
        ['input_images', 'feature_fusion/score_map/Sigmoid', 'feature_fusion/concat_3']
    )
    with tf.gfile.GFile(frozen_model_path, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
