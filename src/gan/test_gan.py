"""Test GAN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow as tf
from common.bio.amino_acid import sequences_to_fasta
from common.bio.sequence import Sequence

from gan.models import get_model
from gan.parameters import get_flags
from gan.documentation import setup_logdir, get_properties
from gan.protein.helpers import convert_to_acid_ids
from common.model.ops import slerp
import numpy as np
from tensorflow.python.training.monitored_session import ChiefSessionCreator, MonitoredSession

slim = tf.contrib.slim
tfgan = tf.contrib.gan

flags.DEFINE_integer('n_seqs', 21, 'Number of sequences to be generated')
flags.DEFINE_boolean('use_cpu', True, 'Flags to determine whether to use CPU or not')
FLAGS = get_flags()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.use_cpu:
        with tf.device('cpu:0'):
            interpolate()
    else:
        interpolate()


def interpolate():
    properties = get_properties(FLAGS)
    logdir = setup_logdir(FLAGS, properties)
    noise = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.z_dim])
    model = get_model(FLAGS, properties, logdir, noise)
    generated_seqs = get_generated_seqs(model)
    session_creator = ChiefSessionCreator(master='', checkpoint_filename_with_path=tf.train.latest_checkpoint(logdir))
    seqs = []
    with MonitoredSession(session_creator=session_creator, hooks=None) as session:
        noise1 = np.random.uniform(-1, 1, FLAGS.z_dim)
        noise2 = np.random.uniform(-1, 1, FLAGS.z_dim)
        n = np.stack([slerp(ratio, noise1, noise2) for ratio in np.linspace(0, 1, FLAGS.batch_size)])
        results, d_scores = session.run([generated_seqs, model.discriminator_fake], feed_dict={noise: n})
        for i in range(FLAGS.batch_size):
            seqs.append(Sequence(id=i, seq=results[i], d_score=d_scores[i]))
        print(sequences_to_fasta(seqs, properties['class_mapping'], escape=False, strip_zeros=True))


def get_generated_seqs(model):
    if FLAGS.one_hot:
        generated_seqs = tf.squeeze(tf.argmax(model.fake_x, axis=-1))
    else:
        generated_seqs = convert_to_acid_ids(model.fake_x)
    return generated_seqs


if __name__ == '__main__':
    tf.app.run()
