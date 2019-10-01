import json
import os
import time

import numpy as np

import tensorflow as tf
from termcolor import colored


def get_model_overview(model_directory):
    path = os.path.join(model_directory, "README.md")
    with open(path, 'r') as f:
        overview = f.read()
    return overview


def get_hyper_parameters(flags, properties):
    path = os.path.join(os.path.dirname(__file__), 'templates', 'HYPERPARAMETERS.md')
    with open(path, 'r') as f:
        hyper_parameters = f.read()
    input_w = properties["seq_length"] if "protein" in flags.dataset else properties["image_width"]
    input_h = flags.embedding_height if "protein" in flags.dataset else properties["image_height"]
    hyper_parameters = hyper_parameters.format(
        # Header
        flags.model_type, flags.dataset,
        # General
        flags.batch_size,
        flags.beta1,
        flags.beta2,
        input_h, input_w,
        flags.loss_type,
        flags.pooling,
        flags.dilation_rate,
        flags.one_hot if hasattr(flags, "one_hot") else False,
        # Discriminator
        flags.architecture,
        flags.discriminator_learning_rate,
        flags.d_step,
        flags.df_dim,
        # Generator
        flags.architecture,
        flags.generator_learning_rate,
        flags.g_step,
        flags.gf_dim,
        flags.z_dim)
    return hyper_parameters


def get_gan_summary():
    path = os.path.join(os.path.dirname(__file__), 'templates', 'MODEL.md')
    with open(path, 'r') as f:
        model = f.read()
    discriminator_layers = get_trainable_layers("model/Discriminator")
    discriminator_layers_string = get_layers_string(discriminator_layers)
    generator_layers = get_trainable_layers("model/Generator")
    generator_layers_string = get_layers_string(generator_layers)
    model = model.format(get_all_parameters(), get_trainable_parameters(),
                         get_subset_trainable_parameters("Discriminator"),
                         get_subset_trainable_parameters("Generator"),
                         "## Discriminator",
                         discriminator_layers_string,
                         "## Generator", generator_layers_string)
    return model


def get_layers_string(layers):
    layers_string = ""
    for layer in layers:
        layers_string += "| {} | \n".format(layer)
    return layers_string


def add_gan_scalars(d_learning_rate, g_learning_rate, d_loss, d_loss_fake, d_loss_real, g_loss,
                    discriminator_real, discriminator_fake):
    tf.summary.scalar('d_loss', d_loss, family="1_loss")
    tf.summary.scalar('g_loss', g_loss, family="1_loss")
    tf.summary.scalar('d_loss_real', tf.reduce_mean(d_loss_real), family="2_loss_component")
    tf.summary.scalar('d_loss_fake', tf.reduce_mean(d_loss_fake), family="2_loss_component")
    tf.summary.scalar('d_real', tf.reduce_mean(discriminator_real), family="3_discriminator_values")
    tf.summary.scalar('d_fake', tf.reduce_mean(discriminator_fake), family="3_discriminator_values")
    tf.summary.scalar('d_lr', d_learning_rate, family="4_learning_rate")
    tf.summary.scalar('g_lr', g_learning_rate, family="4_learning_rate")


def add_model_metadata(logdir, directory_path, flags, properties):
    summary_writer = tf.summary.FileWriter(logdir)
    meta = tf.SummaryMetadata()
    meta.plugin_data.plugin_name = "text"
    summary = tf.Summary()
    overview = get_model_overview(directory_path)
    hyper_params = get_hyper_parameters(flags, properties)
    gan_summary = get_gan_summary()

    write_to_file(logdir, overview, hyper_params, gan_summary, flags)
    summary.value.add(tag="1_Overview", metadata=meta, tensor=tf.make_tensor_proto(overview, dtype=tf.string))
    summary.value.add(tag="2_Hyperparameters", metadata=meta,
                      tensor=tf.make_tensor_proto(hyper_params, dtype=tf.string))
    summary.value.add(tag="3_Model", metadata=meta, tensor=tf.make_tensor_proto(gan_summary, dtype=tf.string))
    summary_writer.add_summary(summary)
    summary_writer.flush()
    summary_writer.close()
    tf.logging.info("Saved meta data in {}".format(logdir.replace("\\", "\\\\")))


def write_to_file(logdir, overview, hyper_params, gan_summary, flags):
    with open(os.path.join(logdir, "run_{}.md".format(time.strftime("%Y_%m_%d_%H_%M_%S"))), "w") as text_file:
        all_info = "\r\n".join([overview, hyper_params, gan_summary, "FLAGS", flags.flags_into_string()])
        print(all_info, file=text_file)


def add_image_grid(name, show_num, images_to_display, image_shape, grid_shape):
    tf.summary.image(name, tf.contrib.gan.eval.image_grid(images_to_display[:show_num],
                                                          grid_shape=grid_shape,
                                                          image_shape=(image_shape[0], image_shape[1]),
                                                          num_channels=image_shape[2]))


def print_run_meta_data(flags):
    print('Running model {} with {} data set'.format(colored(flags.model_type, 'red', attrs=['bold']),
                                                     colored(flags.dataset, 'red', attrs=['bold'])))
    print('Batch size: {}'.format(colored(flags.batch_size, attrs=['bold'])))
    print('Learning rates used: discriminator {} generator {} (Beta: {})'.format(
        colored(flags.discriminator_learning_rate, attrs=['bold']),
        colored(flags.generator_learning_rate, attrs=['bold']),
        colored(flags.beta1, attrs=['bold'])))

    print('Discriminator {} dim. Generator {} dim'.format(colored(flags.gf_dim, attrs=['bold']),
                                                          colored(flags.df_dim, attrs=['bold'])))
    print("D_step: {} G step: {}".format(colored(flags.d_step, attrs=['bold']), colored(flags.g_step, attrs=['bold'])))
    print('')
    print(colored('!!! STARTING TRAINING !!!', attrs=['bold']))
    print('')


def print_model_parameters():
    all_parameters = get_all_parameters()
    trainable_parameters = get_trainable_parameters()
    print("All parameters: {} (out of them {} are trainable)".format(all_parameters, trainable_parameters))
    d_params = get_subset_trainable_parameters("Discriminator")
    g_params = get_subset_trainable_parameters("Generator")
    print("Discriminator {} trainable parameters. Generator trainable parameters {}".format(d_params, g_params))


def get_subset_trainable_parameters(prefix):
    return np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()
                   if prefix in x.name])


def get_trainable_parameters():
    trainable_parameters = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()])
    return trainable_parameters


def get_all_parameters():
    all_parameters = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.global_variables()])
    return all_parameters


def print_trainable_layers():
    [print("{}{}".format(x.name, x.shape)) for x in tf.trainable_variables() if "LayerNorm" not in x.name]


def get_trainable_layers(prefix=""):
    return ["{} | {}".format(x.name.replace(prefix + "/", ""), x.shape) for x in tf.trainable_variables()
            if x.name.startswith(prefix) and "LayerNorm" not in x.name]


def print_model_summary():
    print_model_parameters()
    print("")
    print_trainable_layers()


def get_gan_vars():
    t_vars = tf.trainable_variables()
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model/Discriminator')
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model/Generator')
    for x in d_vars:
        assert x not in g_vars
    for x in g_vars:
        assert x not in d_vars
    for x in t_vars:
        if not x.name.endswith("acid_embeddings:0") and not x.name.startswith("model/convert_to_indices"):
            assert x in g_vars or x in d_vars, x.name
    all_vars = t_vars

    print("********** Model **********")
    print_model_parameters()
    pprint_variables(d_vars, "Discriminator")
    pprint_variables(g_vars, "Generator")

    return all_vars, d_vars, g_vars


def pprint_variables(vars, name):
    print("********** {} **********".format(name))
    [print("{:<40}| {:<20}| {}".format(x.name.replace("model/{}/".format(name), ""), str(x.shape),
                                       np.sum(np.product([xi.value for xi in x.get_shape()])))) for x in vars]


def get_properties(flags):
    path = os.path.join(flags.data_dir, flags.dataset.replace("\\", os.sep), flags.properties_file)
    with open(path) as json_data_file:
        properties = json.load(json_data_file)
    return properties


def setup_logdir(flags, properties):
    if "protein" in flags.dataset :
        one_hot = "one_hot" if flags.one_hot else "embedding"
        input_size = "{}x{}_{}".format(flags.embedding_height,  properties["seq_length"], one_hot)
    else:
        input_size = "{}x{}x{}".format(properties["image_height"], properties["image_width"], properties["num_channels"])

    model_dir = '%s_k_%sx%s_d_%s_%s' % (flags.name, flags.kernel_height, flags.kernel_width, flags.dilation_rate,
                                        flags.pooling)
    batch_size = "batch_size={}".format(flags.batch_size)

    dim = "d_dim_{}_g_dim_{}".format(flags.df_dim, flags.gf_dim)
    logdir = os.path.join(flags.weights_dir, flags.dataset.replace("\\", os.sep), flags.model_type, flags.architecture,
                          input_size, flags.loss_type, batch_size, dim, model_dir)

    tf.gfile.MakeDirs(logdir)
    tf.logging.info("Results will be saved in: {}".format(logdir.replace("\\", "\\\\")))
    print("")
    return logdir


def print_protein_values(val, score):
    print(np.array_str(val[0], precision=5, suppress_small=True, max_line_width=1000))
    print("Score is {}. Max {} and Min {} ".format(score, np.max(val), np.min(val)))
    print("Mean {} and Std {} ".format(np.mean(val), np.std(val)))
    print("Axis =1: Max {} and Min {} ".format(np.max(val, axis=1), np.min(val, axis=1)))
    return "DONE"