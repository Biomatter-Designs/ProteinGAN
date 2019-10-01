import os
import subprocess

from Bio import SeqIO
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def blast_seq(input_seq, only_first_match=False, alignments=500, descriptions=500, hitlist_size=50):
    """Returns BLAST results for given sequence as well as list of sequence titles

    Args:
      input_seq: protein sequence as string
      only_first_match: flag to return only first match (Default value = False)
      alignments: max number of aligments from BLAST (Default value = 500)
      descriptions: max number of descriptions to show (Default value = 500)
      hitlist_size: max number of hits to return. (Default value = 50)

    Returns:
      list of alignments as well as list of titles of sequences in the alignment results

    """
    seq = input_seq.replace('0', '')
    to_display, all_titles = [], []
    try:
        blast_record = get_blast_record(seq, alignments, descriptions, hitlist_size)
        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                alignment_data = get_alignment_data(alignment, hsp)
                to_display.append(alignment_data)
                all_titles.append(alignment.title)
                if only_first_match:
                    break
    except Exception as e:
        print("Unexpected error when calling NCBIWWW.qblast:", str(e))
        to_display.append("Error!")
    return to_display, all_titles


def get_blast_record(seq, alignments, descriptions, hitlist_size):
    """Calls  NCBI's QBLAST server or a cloud service provider to get alignment results

    Args:
      alignments: max number of aligments from BLAST
      descriptions: max number of descriptions to show
      hitlist_size: max number of hits to return
      seq: protein sequence as string

    Returns:
      single Blast record

    """
    result_handle = NCBIWWW.qblast(program="blastp", database="nr", alignments=alignments,
                                   descriptions=descriptions,
                                   hitlist_size=hitlist_size, sequence=seq)
    blast_record = NCBIXML.read(result_handle)
    return blast_record


def get_alignment_data(alignment, hsp):
    """Formats aligment result

    Args:
      alignment: aligment info from BLAST
      hsp: HSP info

    Returns:
      formatted alignment output

    """
    return "****Alignment**** \nSequence: {} \nLength: {} | Score: {} | e value: {} | identities: {} \n{} \n{} \n{} \n".format(
        alignment.title, alignment.length, hsp.score, hsp.expect, hsp.identities, hsp.query, hsp.match, hsp.sbjct)


def get_local_blast_results(data_dir, db_path, fasta):
    query_path = os.path.join(data_dir, "fasta.fasta")
    with open(query_path, "w+") as f:
        f.write(fasta)

    # TODO: Enzyme class
    blastp = subprocess.Popen(
        ['blastp', '-db', db_path, "-max_target_seqs", "1", "-outfmt", "10 qseqid score evalue pident",
         "-matrix", "BLOSUM45", "-query", query_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    results, err = blastp.communicate()
    return parse_blast_results(results.decode()), err.decode()


def parse_blast_results(results):
    """
    Parses Blast results
    Args:
        results: Decoded output from blastp
    Returns:
        a dictonary where key is qseqid, value is score evalue pident values
    """
    parsed = {}
    for line in results.split(os.linesep):
        parts = line.split(",")
        parsed[parts[0]] = parts[1:]

    return parsed


def write_fasta(data, path, sequence_column="sequence"):
    """
    Store sequences in fasta format
    Args:
        data: data to be stored in dataframe format
        path: location of where file should be saved
        sequence_column: a column of dataframe which contains sequence.

    Returns:
        Stores fasta file
    """
    with open(path, 'w') as f_out:
        for row in data.iterrows():
            seq = Seq(row[1][sequence_column])
            seq_record = SeqRecord(seq, id=str(row[0]), description=row[1]["id"] + "_" + str(row[1]["EC number"]))
            r = SeqIO.write(seq_record, f_out, 'fasta')
            if r != 1: print('Error while writing sequence:  ' + seq_record.id)


def update_sequences_with_blast_results(parsed_results, sequences):
    """
    Parses results from blasp into separate arrays
    Args:
        parsed_results: Parsed results from blastp
        sequences: sequences used in blastp

    Returns:
        Returns lists of sequences, e.values, similarities scores and identities
    """
    similarities, evalues, identity = [], [], []
    for sequence in sequences:
        if str(sequence.id) in parsed_results:
            result = parsed_results[str(sequence.id)]
            sequence.similarity = float(result[0])
            sequence.evalue = float(result[1])
            sequence.identity = float(result[2])
            similarities.append(sequence.similarity)
            evalues.append(sequence.evalue)
            identity.append(sequence.identity)
    return sequences, evalues, similarities, identity
