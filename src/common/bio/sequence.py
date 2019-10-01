import os

from common.bio.constants import ID_TO_AMINO_ACID


class Sequence(object):

    """
    Class to store sequences
    """

    def __init__(self, id, seq, enzyme_class=None, label=None, d_score=None, similarity=None, evalue=None, identity=None):
        self.id = id
        self.sequence = seq
        self.enzyme_class = enzyme_class
        self.label = label
        self.d_score = d_score
        self.similarity = similarity
        self.evalue = evalue
        self.identity = identity

    def is_sequence_in_string(self, sequence):
        return isinstance(sequence, str)

    def convert_to_string(self, sequence):
        if self.is_sequence_in_string(sequence):
            return sequence
        else:
            return "".join([ID_TO_AMINO_ACID[i] for i in sequence])

    def get_seq_in_fasta(self, id_to_enzyme_class, escape=False, strip_zeros=False):

        sequence = self.convert_to_string(self.sequence)
        if strip_zeros:
            sequence = sequence.replace("0", "")
        header = ""

        if self.label is not None:
            header = "class: {}".format(id_to_enzyme_class[str(self.label)])
        if self.d_score is not None:
            header = "{} Discriminator score: {}".format(header, self.d_score)
        if self.similarity is not None:
            header = "{} Similarity: {}".format(header, self.similarity)
        if self.evalue is not None:
            header = "{} E.value: {}".format(header, self.evalue)
        if self.identity is not None:
            header = "{} Identity: {}".format(header, self.identity)
        if escape:
            prefix = "\>"
        else:
            prefix = ">"

        return "{}{} {} {}{}".format(prefix, self.id, header, os.linesep, sequence)
