import threading
import tensorflow as tf
from common.bio.amino_acid import get_protein_sequences


class BlastSummary(threading.Thread):
    def __init__(self, summary_writer, global_step, fake_seq, labels, id_to_enzyme_class, running_mode, d_scores):
        self._summary_writer = summary_writer
        self.fake_seq = fake_seq
        self.labels = labels
        self.global_step = global_step
        self.id_to_enzyme_class = id_to_enzyme_class
        self.running_mode = running_mode
        self.d_scores = d_scores
        self.strip_zeros = False
        threading.Thread.__init__(self)

    def run(self):
        try:
            self.run_blast()
        except Exception as e:
            tf.logging.warning("Unexpected error in BlastSummary thread:", str(e))

    def add_summary(self, text):
        meta = tf.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary = tf.Summary()
        text_tensor = tf.make_tensor_proto(text, dtype=tf.string)
        summary.value.add(tag="BLAST_" + self.running_mode, metadata=meta, tensor=text_tensor)
        self._summary_writer.add_summary(summary, self.global_step)

    def add_scalar(self, name, scalar):
        summary = tf.Summary(value=[tf.Summary.Value(tag=name,
                                           simple_value=scalar)])
        self._summary_writer.add_summary(summary, self.global_step)

    def get_protein_sequences(self):
        return get_protein_sequences(self.fake_seq, self.labels, self.d_scores)