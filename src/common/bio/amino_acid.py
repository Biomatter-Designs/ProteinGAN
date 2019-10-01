import os
from collections import Counter

from common.bio.sequence import Sequence
from Bio.SeqIO.FastaIO import SimpleFastaParser
from common.bio.constants import ID_TO_AMINO_ACID, AMINO_ACID_TO_ID, NON_STANDARD_AMINO_ACIDS
import pandas as pd
import numpy as np


def fasta_to_numpy(path, length):
    """

    Args:
        path: of the fasta file
        separator: used in title of fasta file entry

    Returns: numpy array of sequences

    """
    with open(path) as fasta_file:
        sequences = []
        for title, sequence in SimpleFastaParser(fasta_file):
            sequence = sequence[:length]
            to_pad = length - len(sequence)
            sequence = sequence.rjust(len(sequence) - (to_pad // 2), '0')
            sequence = sequence.ljust(length, '0')
            if len(sequence) < length:
                print(sequence.rjust(to_pad // 2, '0'))
                print(to_pad, to_pad//2, length-len(sequence))
            np_seq = np.asarray([AMINO_ACID_TO_ID[a] for a in sequence])
            sequences.append(np_seq)
        return np.stack(sequences, axis= 0)

def from_amino_acid_to_id(data, column):
    """Converts sequences from amino acid to ids

    Args:
      data: data that contains amino acid that need to be converted to ids
      column: a column of the dataframe that contains amino acid that need to be converted to ids

    Returns:
      array of ids

    """
    return data[column].apply(lambda x: [AMINO_ACID_TO_ID[c] for c in x])


def from_id_from_amino_acid(data, column):
    """Converts sequences from ids to amino acid characters

    Args:
      data: data that contains ids that need to be converted to amino acid
      column: a column of the dataframe that contains ids that need to be converted to amino acid

    Returns:
      array of amino acid

    """
    return [[ID_TO_AMINO_ACID[id] for id in val] for index, val in data[column].iteritems()]


def filter_non_standard_amino_acids(data, column):
    """

    Args:
      data: dataframe containing amino acid sequence
      column: a column of dataframe that contains amino acid sequence

    Returns:
      filtered data drame

    """

    data = data[~data[column].str.contains("|".join(NON_STANDARD_AMINO_ACIDS))]

    return data


def get_protein_sequences(sequences, labels=None, d_scores=None):
    """

    Args:
      sequences: Protein sequences
      id_to_enzyme_class: a dictionary to get enzyme class from its id
      labels: Ids  of Enzyme classes (Default value = None)

    Returns:
      array of Sequence objects
    """
    seqs = []
    for index, seq in enumerate(sequences):
        label = None if labels is None else labels[index]
        d_score = None if d_scores is None else d_scores[index]
        seqs.append(Sequence(index, seq, label=label, d_score=d_score))
    return seqs


def numpy_seqs_to_fasta(sequences, id_to_enzyme_class, labels=None, d_scores=None, strip_zeros=False):
    """

    Args:
      sequences: Protein sequences
      id_to_enzyme_class: a dictionary to get enzyme class from its id
      labels: Ids  of Enzyme classes (Default value = None)
      d_scores: Values of discriminator (Default value = None)
      strip_zeros: Flag to determine if special characters needs to be escape. Applicable for text in tersorboard
    Returns:
      array of strings with sequences and additional information

    """
    seqs = get_protein_sequences(sequences, labels, d_scores)
    return sequences_to_fasta(seqs, id_to_enzyme_class, True, strip_zeros)


def sequences_to_fasta(sequences, id_to_enzyme_class, escape=True, strip_zeros=False):
    """

    Args:
      sequences: a list of Sequences object
      id_to_enzyme_class: a dictionary to get enzyme class from its id
      labels: Ids  of Enzyme classes (Default value = None)
      escape: a flag to determine if special characters needs to be escape. Applicable for text in tersorboard
      strip_zeros: a flag that determines whether zeros are removed from sequences
    Returns:
      string with sequences and additional information that mimics fasta format

    """
    return os.linesep.join([seq.get_seq_in_fasta(id_to_enzyme_class, escape, strip_zeros) for seq in sequences])


def print_protein_seq(sequences, id_to_enzyme_class, labels=None, d_scores=None):
    """

    Args:
      sequences: Protein sequences
      id_to_enzyme_class: a dictionary to get enzyme class from its id
      labels: Ids  of Enzyme classes (Default value = None)
      d_scores: Values of discriminator (Default value = None)

    Returns:
      Signal for DONE

    """
    print("\n".join(numpy_seqs_to_fasta(sequences, id_to_enzyme_class, labels, d_scores)))
    return "DONE"


def fasta_to_pandas(path, separator=";"):
    """

    Args:
        path: of the fasta file
        separator: used in title of fasta file entry

    Returns: pandas dataframe with 3 columns (id, title, sequence)

    """
    with open(path) as fasta_file:
        identifiers, sequences, titles = [], [], []
        for title, sequence in SimpleFastaParser(fasta_file):
            title_parts = title.split(separator, 1)
            identifiers.append(title_parts[0])  # First word is ID
            titles.append("|".join(title_parts[1:]))
            sequences.append(sequence)
        return pd.DataFrame({"id": identifiers, "title": titles, "sequence": sequences})


def fasta_to_numpy(path, length):
    """

    Args:
        path: of the fasta file
        separator: used in title of fasta file entry

    Returns: numpy array of sequences

    """
    with open(path) as fasta_file:
        sequences = []
        for title, sequence in SimpleFastaParser(fasta_file):
            sequence = sequence[:length]
            to_pad = length - len(sequence)
            sequence = sequence.rjust(len(sequence) - (to_pad // 2), '0')
            sequence = sequence.ljust(length, '0')
            if len(sequence) < length:
                print(sequence.rjust(to_pad // 2, '0'))
                print(to_pad, to_pad//2, length-len(sequence))
            np_seq = np.asarray([AMINO_ACID_TO_ID[a] for a in sequence])
            sequences.append(np_seq)
        return np.stack(sequences, axis= 0)


def generate_random_seqs(data, column='sequence', n_seqs=1000):
    """

    Args:
        data: Dataframe that contains sequences
        column: a name of the column which contains sequences

    Returns:
        Randomly generated sequences based on frequency of each element

    """
    results = Counter(data[column].str.cat())
    counts = [i[1] for i in sorted(results.items())]
    prop = np.asarray(counts) / sum(list(counts))
    lengths = data.sequence.str.len().sample(n_seqs).values + int(np.random.normal(scale=3))
    seqs = []
    for i in range(n_seqs):
        r = np.random.choice(np.arange(1, 21), p=prop, size=lengths[i])
        seq = ">R_{}\nM".format(i)
        for a in r:
            seq = seq + ID_TO_AMINO_ACID[a]
        seqs.append(seq)
    return seqs
