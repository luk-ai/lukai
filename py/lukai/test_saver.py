import tensorflow as tf
import tempfile
import shutil

from lukai import saver

def test_saver():
    dir = tempfile.mkdtemp("test_saver")

    x = tf.placeholder(tf.float32, [None, 1], name="x")
    y_ = tf.placeholder(tf.float32, [None, 1], name="y_")

    b = tf.Variable(tf.zeros([1]), name="b")
    w = tf.Variable(tf.zeros([1, 1]), name="w")

    y = tf.identity(w * x + b, name="y")

    loss = tf.reduce_sum((y - y_) * (y - y_))
    train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss, name="train")

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    saver.save(sess, target="{}/model.tar.gz".format(dir))

    shutil.rmtree(dir)

