import re

from Bio import Entrez
from bio.blast import blast_seq
import tensorflow as tf
from gan.protein.blast_summary import BlastSummary


class GlobalBlastSummary(BlastSummary):
    def __init__(self, config, summary_writer, global_step, fake_seq, labels, id_to_enzyme_class, n_examples=2,
                 running_mode="train", d_scores=None):
        self.n_examples = n_examples
        super(BlastSummary, self).__init__(summary_writer, global_step, fake_seq, labels, id_to_enzyme_class,
                                           running_mode, d_scores)

    def run_blast(self):
        tf.logging.info("Running Global Blast thread for step {}".format(self.global_step))
        human_readable_fake = self.get_protein_sequences()[:self.n_examples]
        text = self.get_global_blast_results(human_readable_fake)
        self.add_summary(text)

    def get_global_blast_results(self, fake_seqs):
        info = []
        for fake_seq in fake_seqs:
            to_display, all_titles = blast_seq(fake_seq.convert_to_string(), alignments=3, descriptions=3,
                                               hitlist_size=3)
            classes_string = self.get_enzyme_classes(all_titles)
            fasta_entry = fake_seq.get_seq_in_fasta(self.id_to_enzyme_class, True)
            line = "{} \n \n{} \nEnzyme classes: \n{} \n ".format(fasta_entry, "\r\n".join(to_display), classes_string)
            print(line.replace("\\", ""))
            info.append(line)
        return info

    def get_enzyme_classes(self, all_titles):
        matched = re.findall('gi\|(.+?)\|', "".join(all_titles))
        classes = []
        if len(matched) > 0:
            records = self.fetch_protein_data(matched)
            self.parse_enzyme_classes(classes, records)
        classes_string = ", ".join(classes) if len(classes) > 0 else "Not found"
        return classes_string

    def fetch_protein_data(self, matched):
        try:
            ids_to_search = ",".join(matched)
            handle = Entrez.efetch(db="Protein", id=ids_to_search, retmode="xml")
            records = Entrez.read(handle)
            handle.close()
        except Exception as e:
            print("Error when calling Entrez", str(e))
            records = []
        return records

    def parse_enzyme_classes(self, classes, records):
        for record in records:
            for e in record["GBSeq_feature-table"]:
                if e['GBFeature_key'] == "Protein":
                    for ee in e["GBFeature_quals"]:
                        if ee['GBQualifier_name'] == "EC_number":
                            classes.append(ee["GBQualifier_value"])
