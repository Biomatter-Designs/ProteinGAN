import os

import common.model.utils_ori as utils
import numpy as np
import tensorflow as tf
from common.bio.amino_acid import numpy_seqs_to_fasta

from common.bio.constants import get_lesk_color_mapping
from gan.documentation import add_image_grid
from gan.protein.custom_scalars import add_custom_scalar
from gan.protein.helpers import convert_to_acid_ids, REAL_PROTEINS, ACID_EMBEDDINGS_SCOPE, ACID_EMBEDDINGS, \
    FAKE_PROTEINS, CLASS_MAPPING, SEQ_LENGTH, get_shape, LABELS, NUM_AMINO_ACIDS, get_file
from common.model import ops
from common.model.ops import pad_up_to, gelu



class Protein(object):

    def __init__(self, flags, properties, logdir):
        self.config = flags
        self.classes = properties[CLASS_MAPPING]
        self.num_classes = len(self.classes)
        self.width = properties[SEQ_LENGTH]
        self.embedding_height = self.config.embedding_height
        self.embeddings, self.embeddings_variation = self.get_embedding_data(self.embedding_height, flags)
        self.reactions = self.get_reaction_data(flags)
        self.acid_embeddings = None
        self.shape = get_shape(flags,properties)
        add_custom_scalar(logdir)

    def get_reaction_data(self, flags):
        filename = flags.running_mode + "_reactions.npy"
        reaction_path = os.path.join(flags.data_dir, flags.dataset.replace("\\", os.sep), filename)
        try:
            reactions = np.load(reaction_path)
        except Exception as e:
            tf.logging.warn("Reaction file could not be loaded: " + e.__str__())
            reactions = []
        return reactions

    def get_embedding_data(self, embedding_height, flags):
        embeddings, embeddings_variation = None, None
        if not (self.config.already_embedded or self.config.one_hot):
            name = flags.embedding_name
            embeddings = get_file("{}_{}.npy".format(name, embedding_height), flags)
            embeddings_variation = get_file("{}_variation_{}.npy".format(name, embedding_height), flags)
        return embeddings, embeddings_variation

    def prepare_real_data(self, real_x, labels):
        real_x = tf.reshape(real_x, [self.config.batch_size, self.width], name=REAL_PROTEINS)
        reactions = self.get_reactions(labels)
        labels = tf.identity(tf.squeeze(labels), name=LABELS)
        if self.config.one_hot:
            real_x = self.convert_real_to_one_hot(real_x)
            real_x = tf.expand_dims(real_x, 1)
        elif not self.config.already_embedded:
            real_x = self.get_embedded_seqs(real_x)
        else:
            real_x = tf.reshape(real_x, [self.config.batch_size, self.width, self.embedding_height])
            noise = tf.random_normal(shape=tf.shape(real_x), mean=0.0, stddev=0.008, dtype=tf.float32)
            real_x = tf.clip_by_value(real_x + self.config.noise_level * noise, -1, 1)
            real_x = tf.expand_dims(real_x, 1)
        return real_x, (labels, reactions)

    def display_real_data(self, real_x, labels, show_num, d_scores=None):
        if self.config.running_mode == "train":
            if self.config.already_embedded:
                real_x = self.convert_to_indices(real_x)
            self.display_metrics_about_protein(real_x, "real")
            real_to_display = self.convert_real_to_one_hot(real_x)
            self.display_protein(real_to_display, show_num, "real", self.width)
            self.add_proteins_to_tensorboard(d_scores, "real", labels[0], real_to_display)

    def convert_real_to_one_hot(self, real_x): # Label smoothing?
        real_to_display = tf.one_hot(real_x, 21, axis=1)
        real_to_display = tf.transpose(real_to_display, perm=[0, 2, 1])
        return real_to_display

    def get_embedded_seqs(self, real_x):
        if self.config.static_embedding:
            if self.acid_embeddings is None:
                with tf.variable_scope(ACID_EMBEDDINGS_SCOPE):
                    self.acid_embeddings = tf.get_variable(ACID_EMBEDDINGS,
                                                           shape=[NUM_AMINO_ACIDS, self.embedding_height],
                                                           initializer=tf.constant_initializer(self.embeddings),
                                                           trainable=False)
            real_x_emb = tf.nn.embedding_lookup(self.acid_embeddings, real_x)
            real_x_emb = self.add_noise(real_x, real_x_emb)
        else:
            real_x_emb = ops.sn_embedding(real_x, NUM_AMINO_ACIDS, self.embedding_height,
                                          name=ACID_EMBEDDINGS_SCOPE,
                                          embedding_map_name=ACID_EMBEDDINGS)
            real_x_emb.set_shape(self.shape)
        # real_x_emb = tf.transpose(real_x_emb, perm=[0, 2, 1])
        real_x_emb = tf.expand_dims(real_x_emb, 1)
        print("Real", real_x_emb.shape)
        return real_x_emb

    def add_noise(self, real_x, real_x_emb):
        real_x_one_hot = tf.one_hot(real_x, NUM_AMINO_ACIDS, axis=1)

        variation = []
        for embbedding in self.embeddings_variation:
            embbedding_variation = []
            for acid in embbedding:
                # embbedding_variation.append(tf.random_uniform([1], minval=acid[0], maxval=acid[1]))
                embbedding_variation.append(
                    tf.truncated_normal([self.config.batch_size], mean=(acid[0] + acid[1]) / 2.0,
                                        stddev=abs(acid[0] - acid[1]) / 4.0))
            variation.append(embbedding_variation)

        t_variation = tf.squeeze(tf.convert_to_tensor(variation))
        t_variation = tf.transpose(t_variation, perm=[2, 0, 1])
        t_variation = self.config.noise_level * t_variation
        noise = tf.matmul(t_variation, real_x_one_hot)
        noise = tf.transpose(noise, perm=[0, 2, 1])
        # noise_extra = tf.random_normal([self.config.batch_size, self.width, self.config.input_h],
        #                          stddev=0.001, dtype=tf.float32)
        noise_extra = tf.random_uniform([self.config.batch_size, self.width, self.embedding_height],
                                        maxval=0.009, minval=-0.009, dtype=tf.float32)
        real_x_emb = tf.add(real_x_emb, noise) + noise_extra
        return real_x_emb

    def display_fake_data(self, fake_x, labels, show_num, d_scores=None, to_display=True):
        family = "fake"
        if self.config.one_hot:
            fake_to_display = tf.argmax(fake_x, axis=-1)
            fake_to_display = tf.squeeze(fake_to_display, name=FAKE_PROTEINS)
        else:
            if self.config.already_embedded:
                fake_to_display = tf.squeeze(self.convert_to_indices(fake_x), name=FAKE_PROTEINS)
            else:
                fake_to_display, _ = convert_to_acid_ids(fake_x, self.config.batch_size)
        print_fake = tf.print(family, fake_to_display[0], summarize=self.width)
        with tf.control_dependencies([print_fake]):
            if to_display:
                self.display_metrics_about_protein(fake_to_display, family)
                fake_to_display = tf.one_hot(fake_to_display, NUM_AMINO_ACIDS, axis=1)
                fake_to_display = tf.transpose(fake_to_display, [0, 2, 1])
                self.display_protein(fake_to_display, show_num, family, self.width)
                self.add_proteins_to_tensorboard(d_scores, family, labels[0], fake_to_display)

    def display_protein(self, protein, show_num, family, protein_len):
        protein_to_display = self.color_protein(protein, protein_len)
        image_shape = [protein_to_display.shape[1], protein_to_display.shape[2], protein_to_display.shape[3]]
        add_image_grid(family + "_image_grid", show_num, protein_to_display, image_shape, (show_num, 1))

    def add_proteins_to_tensorboard(self, d_scores, family, labels, protein):
        with tf.variable_scope(self.config.running_mode, reuse=True):
            variables = [tf.argmax(tf.squeeze(protein), axis=-1), tf.squeeze(labels), d_scores]
            tf.summary.text(family, tf.py_func(
                lambda vals, labels, d_scores: numpy_seqs_to_fasta(vals, self.classes, labels, d_scores), variables,
                tf.string))

    def display_metrics_about_protein(self, data, family):
        flatten = tf.reshape(data, [-1])
        y, idx, count = tf.unique_with_counts(flatten)
        tf.summary.scalar(family, tf.size(count), family="unique_amino_acids")
        tf.summary.histogram(family, flatten, family="distribution_of_values")

    def color_protein(self, protein, protein_len=128):
        colors = tf.expand_dims(get_lesk_color_mapping(), 0)
        colors = tf.tile(colors, [self.config.batch_size, 1, 10])
        colored = tf.matmul(protein, colors)
        colored = tf.reshape(colored, [colored.shape[0], protein_len, 10, 3])
        colored = tf.transpose(colored, perm=[0, 2, 1, 3])
        return colored

    def get_batch(self, batch_size, config):
        path = os.path.join(config.data_dir, config.dataset.replace("\\", os.sep))
        extract_fn = utils.extract_emb_seq_and_label if config.already_embedded else utils.extract_seq_and_label
        batches = utils.get_batches(extract_fn, path, batch_size, shuffle_buffer_size=config.shuffle_buffer_size,
                                    running_mode=config.running_mode, args=[[self.width], self.config.dynamic_padding])
        return batches

    def get_reactions(self, labels):
        reaction_tensors = []
        for reaction in self.reactions:
            # label = tf.convert_to_tensor(reaction[0])
            r = [pad_up_to(tf.convert_to_tensor(reaction[i]), [self.config.compound_w], self.config.dynamic_padding) for
                 i in range(1, 5)]
            reaction_tensors.append(r)
        reaction_tensors = tf.stack(reaction_tensors)
        return tf.gather_nd(reaction_tensors, tf.expand_dims(labels, axis=1))

    def convert_to_indices(self, input_tensor):
        with tf.variable_scope("convert_to_indices"):
            with tf.variable_scope("bert"):
                input_tensor = tf.reshape(tf.squeeze(input_tensor), [-1, self.embedding_height])
                print(input_tensor.shape)
                output_weights = tf.get_variable(
                    name="embeddings/word_embeddings",
                    shape=[NUM_AMINO_ACIDS + 1, self.embedding_height],
                    initializer=tf.zeros_initializer())
            with tf.variable_scope("cls/predictions"):
                # We apply one more non-linear transformation before the output layer.
                # This matrix is not used after pre-training.
                with tf.variable_scope("transform"):
                    input_tensor = tf.layers.dense(
                        input_tensor,
                        units=self.embedding_height,
                        activation=gelu,
                        kernel_initializer=tf.zeros_initializer())
                    input_tensor = tf.contrib.layers.layer_norm(inputs=input_tensor)

                # The output weights are the same as the input embeddings, but there is
                # an output-only bias for each token.
                output_bias = tf.get_variable(
                    "output_bias",
                    shape=[NUM_AMINO_ACIDS + 1],
                    initializer=tf.zeros_initializer())
                logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)
                logits = tf.reshape(logits, [self.config.batch_size, self.width, NUM_AMINO_ACIDS + 1])
        return tf.squeeze(tf.argmax(logits, axis=2))
