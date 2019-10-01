"""Generate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
from absl import flags
from bio.amino_acid import sequences_to_fasta
from bio.blast import get_local_blast_results, update_sequences_with_blast_results
from bio.sequence import Sequence
from gan.documentation import setup_logdir, get_properties
from gan.models import get_model
from gan.parameters import get_flags
from gan.protein.helpers import convert_to_acid_ids
from tensorflow.python.training.monitored_session import ChiefSessionCreator, MonitoredSession

flags.DEFINE_integer('n_seqs', 21, 'Number of sequences to be generated')
flags.DEFINE_float('stddev', 0.5, 'Standard deviation of noise')
flags.DEFINE_boolean('use_cpu', True, 'Flags to determine whether to use CPU or not')
flags.DEFINE_boolean('blast', False, 'Flags to determine whether to add blast results')
FLAGS = get_flags()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.use_cpu:
        with tf.device('cpu:0'):
            generate_sequences()
    else:
        generate_sequences()


def generate_sequences():
    properties = get_properties(FLAGS)
    logdir = setup_logdir(FLAGS, properties)
    tf.logging.info('Noise will have standard deviation of {}'.format(FLAGS.stddev))
    noise = tf.random.truncated_normal([FLAGS.batch_size, FLAGS.z_dim], stddev=FLAGS.stddev, dtype=tf.float32)
    model = get_model(FLAGS, properties, logdir, noise)
    if FLAGS.one_hot:
        generated_seqs = tf.squeeze(tf.argmax(model.fake_x, axis=-1))
    else:
        generated_seqs = convert_to_acid_ids(model.fake_x)
    seqs = []
    session_creator = ChiefSessionCreator(master='', checkpoint_filename_with_path=tf.train.latest_checkpoint(logdir))
    with MonitoredSession(session_creator=session_creator, hooks=None) as session:
        while True:
            results, step = session.run([generated_seqs, tf.train.get_global_step()], None)
            id = len(seqs)
            for i in range(FLAGS.batch_size):
                seqs.append(Sequence(id=id + i, seq=results[i]))
            if len(seqs) >= FLAGS.n_seqs:
                break
    time_stamp = time.strftime('%H_%M_%S', time.gmtime())
    path = os.path.join(logdir, 'generated_{}_{}.fasta'.format(step, time_stamp))
    fasta = sequences_to_fasta(seqs, properties['class_mapping'], escape=False, strip_zeros=True)
    if FLAGS.blast:
        db_path = os.path.join(FLAGS.data_dir, FLAGS.dataset,
                               FLAGS.blast_db.replace("\\", os.sep) + "_" + FLAGS.running_mode)
        blast_results, err = get_local_blast_results(logdir, db_path, fasta)
        seqs, evalues, similarities, identity = update_sequences_with_blast_results(blast_results, seqs)
        print_stats([("Evalue", evalues), ("BLOMSUM45", similarities), ("Identity", identity)], len(seqs))
        fasta = sequences_to_fasta(seqs, properties['class_mapping'], escape=False, strip_zeros=True)
    with open(path, 'w') as f:
        print(fasta, file=f)
        tf.logging.info('{} sequences stored in {}'.format(len(seqs), path))
    tf.logging.info('Finished evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))


def print_stats(stats, n_seqs):
    for name, data in stats:
        avg = sum(data) / n_seqs
        min_value = min(data)
        max_value = max(data)
        tf.logging.info("{:10s}: AVG: {:.2f} | MIN: {:.2f} | MAX {:.2f}".format(name, avg, min_value, max_value))


if __name__ == '__main__':
    tf.app.run()
