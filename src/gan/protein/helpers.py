import os

import numpy as np
import tensorflow as tf
from gan.parameters import DATASET

ACID_EMBEDDINGS = "acid_embeddings"
ACID_EMBEDDINGS_SCOPE = "acid_emb_scope"
REAL_PROTEINS = "real_proteins"
FAKE_PROTEINS = "fake_proteins"
CLASS_MAPPING = "class_mapping"
LABELS = "labels"
NUM_AMINO_ACIDS = 21
SEQ_LENGTH = "seq_length"


def convert_to_acid_ids(fake_x, batch_size):
    fake_to_display = tf.squeeze(fake_x)
    acid_embeddings = tf.get_variable(ACID_EMBEDDINGS_SCOPE + "/" + ACID_EMBEDDINGS)
    fake_to_display, distances = reverse_embedding_lookup(acid_embeddings, fake_to_display, batch_size)
    fake_to_display = tf.squeeze(fake_to_display, name=FAKE_PROTEINS)
    tf.summary.scalar("FAKE", tf.reduce_mean(distances, name="fake_cosine_d"), family="Cosine_distance")
    return fake_to_display, distances


def reverse_embedding_lookup(acid_embeddings, embedded_sequence, batch_size):
    embedded_sequence = tf.transpose(embedded_sequence, [0, 2, 1])
    acid_embeddings_expanded = tf.tile(tf.expand_dims(acid_embeddings, axis=0), [batch_size, 1, 1])
    emb_distances = tf.matmul(
        tf.nn.l2_normalize(acid_embeddings_expanded, axis=2),
        tf.nn.l2_normalize(embedded_sequence, axis=1))
    indices = tf.argmax(emb_distances, axis=1)
    return indices, tf.reduce_max(emb_distances, axis=1)


def test_amino_acid_embeddings(acid_embeddings, real_x, width):
    print_op_b = tf.print(tf.transpose("REAL_127:", real_x[0], perm=[1, 0])[127, :], summarize=width)
    print_op_e = tf.print(tf.transpose("REAL_0:", real_x[0], perm=[1, 0])[0, :], summarize=width)
    real_x, _ = reverse_embedding_lookup(acid_embeddings, real_x)
    print_op_a = tf.print("RECO!: ", real_x[0], summarize=width)
    with tf.control_dependencies([print_op_b, print_op_e, print_op_a]):
        tf.summary.histogram("test", real_x[0], family="test_reconstruction")


def get_shape(config, properties):
    width = properties[SEQ_LENGTH]
    if config.one_hot:
        return [config.batch_size, 1, width, NUM_AMINO_ACIDS]
    else:
        return [config.batch_size, 1, width, config.embedding_height]


def get_file(filename, flags):
    embedding_path = os.path.join(flags.data_dir, DATASET, filename)
    return np.load(embedding_path)
