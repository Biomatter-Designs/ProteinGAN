from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import platform

from gan.models import get_model, get_specific_hooks
from gan.parameters import get_flags
from gan.documentation import setup_logdir, get_properties
from gan.documentation import print_run_meta_data, add_model_metadata

import tensorflow as tf
from gan.protein.embedding_hook import get_embedding_hook
from tensorflow.contrib.gan import GANTrainOps, GANTrainSteps, gan_train, get_sequential_train_hooks
from tensorflow.python import debug as tf_debug

FLAGS = get_flags()

def main(_, is_test=False, debug_cli=False, debug_ui=False):
    graph = tf.Graph()
    with graph.as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        properties = get_properties(FLAGS)
        # Select model to train
        logdir = setup_logdir(FLAGS, properties)
        noise = tf.random.truncated_normal([FLAGS.batch_size, 128], stddev=0.5, dtype=tf.float32, name='noise')
        model = get_model(FLAGS, properties, logdir, noise)
        print_run_meta_data(FLAGS)
        # Adding all meta data about the model before starting
        add_model_metadata(logdir, os.path.join(os.path.dirname(__file__), FLAGS.model_type), FLAGS, properties)

        # We set allow_soft_placement to be True because Saver for the DCGAN model gets misplaced on the GPU.
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

        hooks = get_hooks(debug_cli, debug_ui)
        model_hooks = get_specific_hooks(FLAGS, logdir, properties)
        if hasattr(FLAGS, "static_embedding") and not FLAGS.static_embedding:
            model_hooks.append(get_embedding_hook(model, FLAGS))

        train_ops = GANTrainOps(generator_train_op=model.g_optim,
                                discriminator_train_op=model.d_optim,
                                global_step_inc_op=model.increment_global_step)
        train_steps = GANTrainSteps(FLAGS.g_step, FLAGS.d_step)

        if is_test:
            return graph
        else:
            with tf.variable_scope('gan_train', reuse=tf.AUTO_REUSE) as scope:
                gan_train(train_ops,
                          get_hooks_fn=get_sequential_train_hooks(train_steps=train_steps),
                          hooks=([tf.train.StopAtStepHook(num_steps=FLAGS.steps)] + hooks + model_hooks),
                          logdir=logdir,
                          save_summaries_steps=FLAGS.save_summary_steps,
                          save_checkpoint_secs=FLAGS.save_checkpoint_sec,
                          config=session_config)


def get_hooks(debug_cli, debug_ui):
    hooks = []
    if debug_cli:
        cli_debug_hook = tf_debug.LocalCLIDebugHook()
        cli_debug_hook.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        hooks.append(cli_debug_hook)
    elif debug_ui:
        debug_host = "{}:5002".format(platform.node())
        hooks.append(tf_debug.TensorBoardDebugHook(debug_host, send_traceback_and_source_code=False))
        print("Debugger is running on {}".format(debug_host))
    return hooks

if __name__ == '__main__':
    tf.app.run()
