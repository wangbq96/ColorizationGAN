import os
import tensorflow as tf
from model import ColorizationGAN


# dir
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint", "models are saved here")
tf.app.flags.DEFINE_string("sample_dir", "./sample", "sample are saved here")
tf.app.flags.DEFINE_string("test_dir", "./test", "test sample are saved here")
tf.app.flags.DEFINE_string("dataset_dir", './img_data', "test sample are saved here")

# params for dataset and environment
tf.app.flags.DEFINE_string("dataset_name", "lsun_bedroom", "name of the dataset")
tf.app.flags.DEFINE_integer("img_size", 256, "size of image")
tf.app.flags.DEFINE_integer("epoch", 1, "# of epoch")
tf.app.flags.DEFINE_integer("batch_size", 16, '# images in batch')
tf.app.flags.DEFINE_string("model_name", "color-cgan", "")

tf.app.flags.DEFINE_float("lr", 0.0002, "initial learning rate for adam")
tf.app.flags.DEFINE_float("beta1", 0.5, "momentum term of adam")
tf.app.flags.DEFINE_float("L1_lambda", 100.0, "weight on L1 term in objective")


tf.app.flags.DEFINE_float("clip_value", 0.01, "")

tf.app.flags.DEFINE_bool("is_train", False, "")

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.test_dir):
        os.makedirs(FLAGS.test_dir)

    with tf.Session() as sess:

        model = ColorizationGAN(sess, config=FLAGS)
        if FLAGS.is_train:
            model.train(FLAGS)
        else:
            model.test(FLAGS)


if __name__ == '__main__':
    tf.app.run()
