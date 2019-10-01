import tensorflow as tf
from gan.documentation import pprint_variables
from gan.protein.protein import ACID_EMBEDDINGS_SCOPE

from tensorflow.contrib.gan import RunTrainOpsHook


def get_embedding_hook(model, config, training_step=1):
    d_optimizer = tf.train.AdamOptimizer(learning_rate=config.discriminator_learning_rate, name='d_opt',
                                         beta1=config.beta1,
                                         beta2=config.beta2)

    emb_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model/' + ACID_EMBEDDINGS_SCOPE)
    [tf.summary.histogram(x.name.replace("model/", ""), x, family="Weights_{}".format(ACID_EMBEDDINGS_SCOPE))
     for x in emb_vars if "beta" not in x.name and "gamma" not in x.name]
    pprint_variables(emb_vars, ACID_EMBEDDINGS_SCOPE)
    loss = model.d_loss + model.g_loss
    emb_optim = d_optimizer.minimize(loss, var_list=emb_vars)

    return RunTrainOpsHook(emb_optim, training_step)
