import os
import pickle
import random

import numpy as np


def load_pickle(path):
    try:
        import dill as loader
    except ImportError:
        loader = pickle
    with open(path, "rb") as handle:
        return loader.load(handle)


def ordered_words(vocabulary):
    mapping = vocabulary.idx2word
    return [mapping[i] for i in range(len(mapping))]


def load_ddi(data_dir):
    ddi = np.asarray(load_pickle(os.path.join(data_dir, "ddi_A_final.pkl")), dtype=np.float32)
    ddi = np.maximum(ddi, ddi.T)
    np.fill_diagonal(ddi, 0.0)
    return ddi


def load_raw(data_dir):
    records = load_pickle(os.path.join(data_dir, "records_final.pkl"))
    voc = load_pickle(os.path.join(data_dir, "voc_final.pkl"))
    ddi = load_ddi(data_dir)
    return (
        records,
        ordered_words(voc["diag_voc"]),
        ordered_words(voc["pro_voc"]),
        ordered_words(voc["med_voc"]),
        ddi,
    )


def split_patients(records, seed, ratios=(0.7, 0.1, 0.2)):
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    n_train = int(len(order) * ratios[0])
    n_val = int(len(order) * ratios[1])
    train = [records[i] for i in order[:n_train]]
    val = [records[i] for i in order[n_train:n_train + n_val]]
    test = [records[i] for i in order[n_train + n_val:]]
    return train, val, test
