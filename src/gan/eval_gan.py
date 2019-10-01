"""Evaluation of GAN"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf
from absl import flags
from gan.documentation import setup_logdir, get_properties
from gan.models import get_model
from gan.parameters import get_flags
from tensorflow.python.training.monitored_session import ChiefSessionCreator, MonitoredSession

slim = tf.contrib.slim
tfgan = tf.contrib.gan

flags.DEFINE_boolean('use_cpu', True, 'Flags to determine whether to use CPU or not')
FLAGS = get_flags()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.use_cpu:
        with tf.device('cpu:0'):
            raw_results()
    else:
        raw_results()


def raw_results():
    properties = get_properties(FLAGS)
    logdir = setup_logdir(FLAGS, properties)
    noise = tf.random.truncated_normal([FLAGS.batch_size, FLAGS.z_dim], stddev=0.5, dtype=tf.float32)
    model = get_model(FLAGS, properties, logdir, noise)
    raw_generations = tf.squeeze(model.fake_x)
    session_creator = ChiefSessionCreator(master='', checkpoint_filename_with_path=tf.train.latest_checkpoint(logdir))
    with MonitoredSession(session_creator=session_creator, hooks=None) as session:
        results, step = session.run([raw_generations, tf.train.get_global_step()], None)
        time_stamp = time.strftime('%H_%M_%S', time.gmtime())
        path = os.path.join(logdir, 'raw_{}_{}.npz'.format(step, time_stamp))
        with open(path, 'wb') as f:
            np.savez(f,results)


if __name__ == '__main__':
    tf.app.run()
