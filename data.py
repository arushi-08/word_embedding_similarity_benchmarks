import os
import json
import requests
import csv
import nltk
import pandas as pd
import numpy as np
import tensorflow as tf

from constants import STOP, NUM_DOCS, PATH_TO_DOC_FREQUENCIES_FILE


def save_word_similarity_dataset():
    tasks = {
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "MTurk": fetch_MTurk(),
        "RG65": fetch_RG65(),
        "RW": fetch_RW(),
        "SimLex999": fetch_SimLex999(),
        "TR9856": fetch_TR9856()
    }
    for dataname in tasks:
        for items in tasks[dataname]:
            if (isinstance(tasks[dataname][items], list) 
                or isinstance(tasks[dataname][items], np.ndarray)):
                tasks[dataname][items] = [list(i) for i in tasks[dataname]['X']]
    
    with open('all_datasets_2.json', 'w') as f:
        json.dump(tasks, f)


def load_sts_dataset(filename):
    # Loads a subset of the STS dataset into a DataFrame. In particular both
    # sentences and their human rated similarity score.
    sent_pairs = []
    with tf.io.gfile.GFile(filename, "r") as f:
        for line in f:
            ts = line.strip().split("\t")
            sent_pairs.append((ts[5], ts[6], float(ts[4])))
    return pd.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])


def download_and_load_sts_data():
    sts_dataset = tf.keras.utils.get_file(
        fname="Stsbenchmark.tar.gz",
        origin="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz",
        extract=True)

    sts_dev = load_sts_dataset(os.path.join(os.path.dirname(sts_dataset), "stsbenchmark", "sts-dev.csv"))
    sts_test = load_sts_dataset(os.path.join(os.path.dirname(sts_dataset), "stsbenchmark", "sts-test.csv"))

    return sts_dev, sts_test


def download_sick(f): 
    response = requests.get(f).text
    lines = response.split("\n")[1:]
    lines = [l.split("\t") for l in lines if len(l) > 0]
    lines = [l for l in lines if len(l) == 5]
    df = pd.DataFrame(lines, columns=["idx", "sent_1", "sent_2", "sim", "label"])
    df['sim'] = pd.to_numeric(df['sim'])
    return df
    

class Sentence:
    
    def __init__(self, sentence):
        self.raw = sentence
        normalized_sentence = sentence.replace("‘", "'").replace("’", "'")
        self.tokens = [t.lower() for t in nltk.word_tokenize(normalized_sentence)]
        self.tokens_without_stop = [t for t in self.tokens if t not in STOP]


def read_tsv(f):
    frequencies = {}
    with open(f) as tsv:
        tsv_reader = csv.reader(tsv, delimiter="\t")
        for row in tsv_reader: 
            frequencies[row[0]] = int(row[1])
        
    return frequencies

doc_frequencies = read_tsv(PATH_TO_DOC_FREQUENCIES_FILE)
doc_frequencies["NUM_DOCS"] = NUM_DOCS


