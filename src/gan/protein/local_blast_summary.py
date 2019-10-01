import os
import tensorflow as tf
import numpy as np
from common.bio.amino_acid import sequences_to_fasta
from common.bio.blast import get_local_blast_results, update_sequences_with_blast_results
from gan.protein.blast_summary import BlastSummary


class LocalBlastSummary(BlastSummary):
    def __init__(self, config, summary_writer, global_step, fake_seq, labels, id_to_enzyme_class, n_examples=2,
                 running_mode="train", d_scores=None):
        self.config = config
        self.db_path = os.path.join(config.data_dir, config.dataset, config.blast_db.replace("\\", os.sep))
        super(LocalBlastSummary, self).__init__(summary_writer, global_step, fake_seq, labels, id_to_enzyme_class,
                                                running_mode, d_scores)

    def run_blast(self):
        tf.logging.info("Running Local Blast thread for step {}".format(self.global_step))
        self.strip_zeros = True
        sequences = self.get_protein_sequences()
        fasta = sequences_to_fasta(sequences, self.id_to_enzyme_class, escape=False, strip_zeros=True)
        self.print_blast_results(sequences, fasta, "val")
        sequences = self.print_blast_results(sequences, fasta, "train")
        self.add_updated_text_to_tensorboard(sequences)

    def print_blast_results(self, sequences, fasta, type):
        result, err = get_local_blast_results(self._summary_writer.get_logdir(), self.db_path+"_"+type, fasta)
        sequences, evalues, similarities, identities = update_sequences_with_blast_results(result, sequences)
        avg_similarities, s_max = self.get_stats(len(sequences), similarities, "{}/BLOMSUM45".format(type), np.max)
        avg_evalues, e_min = self.get_stats(len(evalues), evalues, "{}/Evalue".format(type), np.min)
        avg_identities, i_max = self.get_stats(len(identities), identities, "{}/Identity".format(type), np.max)
        template = " BLAST {:5s}: BLOMSUM45: {:.2f}({:.2f}) | E.value: {:.3f}({:.3f}) | Identity: {:.2f}({:.2f})"
        tf.logging.info(template.format(type, avg_similarities, s_max, avg_evalues, e_min, avg_identities, i_max))
        return sequences

    def get_stats(self, batch_size, similarities, name, f):
        avg = np.array(similarities).sum() / batch_size
        best_value = f(similarities)
        self.add_scalar("Blast/{}".format(name), avg)
        return avg, best_value

    def add_updated_text_to_tensorboard(self, sequences):
        text = sequences_to_fasta(sequences, self.id_to_enzyme_class, escape=True, strip_zeros=False)
        self.add_summary(text)
