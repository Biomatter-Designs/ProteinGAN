import tensorflow as tf

from gan.gan_base_model import VariableRestorer
from gan.image.image import Image
from gan.protein.blast_hook import BlastHook
from gan.protein.local_blast_summary import LocalBlastSummary
from gan.protein.protein import Protein
from gan.sngan.model import SNGAN
from gan.wgan.model import WGAN


def get_model(flags, properties, logdir, noise):
    if flags.model_type == "sngan":
        if "image" in flags.dataset:
            gan = SNGAN(Image(flags, properties, logdir), noise)
        elif "protein" in flags.dataset:
            gan = SNGAN(Protein(flags, properties, logdir), noise)
        else:
            raise NotImplementedError

    elif flags.model_type == "wgan":
        if flags.dataset == "mnist":
            gan = WGAN(Image(flags, properties, logdir), noise)
        elif "protein" in flags.dataset:
            gan = WGAN(Protein(flags, properties, logdir), noise)
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
    return gan


def get_specific_hooks(flags, logdir, properties):
    hooks = []
    if "protein" in flags.dataset:
        # print("No Blast hook")
        id_to_enzyme_class_dict = properties["class_mapping"]

        hooks.append(BlastHook(LocalBlastSummary,
                               flags,
                               id_to_enzyme_class_dict,
                               every_n_steps=flags.steps_for_blast,
                               output_dir=logdir))
        tensors_to_log = {" Step": "global_step",
                          "Loss (Disc)": "model/d_loss",
                          "Loss (Gen)": "model/g_loss",
                          "Dev (Real)": "model/stddev/real",
                          "Dev (Fake)": "model/stddev/fake"}
        if not flags.one_hot:
            tensors_to_log["Distance"] = "model/fake_cosine_d"
        hooks.append(tf.train.LoggingTensorHook(tensors_to_log, every_n_iter=flags.save_summary_steps))
        if flags.already_embedded:
            hooks.append(VariableRestorer(flags.variable_path, "model/convert_to_indices/"))

    return hooks
