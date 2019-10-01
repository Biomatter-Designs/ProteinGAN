"""Test GAN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import flags

import tensorflow as tf
from common.bio.amino_acid import sequences_to_fasta, fasta_to_numpy
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
flags.DEFINE_string('fasta_path', '/home/asb/Downloads/generated_2493527_13_37_47.fasta', 'A path to a fasta file for which to retrieve discriminator results')
FLAGS = get_flags()

SEQ_LENGTH = "seq_length"

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.use_cpu:
        with tf.device('cpu:0'):
            get_discriminator_results()
    else:
        get_discriminator_results()


def get_discriminator_results():
    properties = get_properties(FLAGS)
    logdir = setup_logdir(FLAGS, properties)
    noise = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.z_dim])
    model = get_model(FLAGS, properties, logdir, noise)
    s1 = [FLAGS.batch_size, properties[SEQ_LENGTH]]
    input = tf.placeholder(dtype=tf.int32, shape=s1)
    data = tf.expand_dims(tf.transpose(tf.one_hot(input, FLAGS.n_seqs, axis=1), [0,2,1]), axis=1)
    s2 = [FLAGS.batch_size]
    labels = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size])
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        d, d_h = model.get_discriminator_result(data, labels, reuse=True)

    fasta_seqs = fasta_to_numpy(FLAGS.fasta_path, properties[SEQ_LENGTH])
    session_creator = ChiefSessionCreator(master='', checkpoint_filename_with_path=tf.train.latest_checkpoint(logdir))
    seqs = []
    with MonitoredSession(session_creator=session_creator, hooks=None) as session:
        for i in range(0, len(fasta_seqs), FLAGS.batch_size):
            print("Processing batch ", i)
            batch = fasta_seqs[i:i+FLAGS.batch_size]
            l = len(batch)
            if l < (FLAGS.batch_size):
                batch = np.vstack([batch, np.zeros([FLAGS.batch_size-l,properties[SEQ_LENGTH]])])
            d_scores, step = session.run([d, tf.train.get_global_step()], feed_dict={input: batch, labels: np.zeros(s2)})
            for j in range(l):
                seqs.append(Sequence(id=j+i, seq=fasta_seqs[j+i], d_score=d_scores[j]))
        fasta = sequences_to_fasta(seqs, properties['class_mapping'], escape=False, strip_zeros=True)
        time_stamp = time.strftime('%H_%M_%S', time.gmtime())
        original_name = os.path.splitext(os.path.basename(FLAGS.fasta_path))[0]
        path = os.path.join(logdir, '{}_scores_{}_{}.fasta'.format(original_name,step, time_stamp))
        with open(path, 'w') as f:
            print(fasta, file=f)
            tf.logging.info('{} sequences stored in {}'.format(len(seqs), path))


def get_generated_seqs(model):
    if FLAGS.one_hot:
        generated_seqs = tf.squeeze(tf.argmax(model.fake_x, axis=-1))
    else:
        generated_seqs = convert_to_acid_ids(model.fake_x)
    return generated_seqs


if __name__ == '__main__':
    tf.app.run()

