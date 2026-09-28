from itertools import combinations

import numpy as np

EPS = 1e-6


def medication_visit_counts(records, num_meds):
    counts = np.zeros(num_meds, dtype=np.float64)
    for patient in records:
        for visit in patient:
            for m in set(visit[2]):
                if 0 <= m < num_meds:
                    counts[m] += 1
    return counts


def cooccurrence_counts(records, num_meds):
    counts = np.zeros(num_meds, dtype=np.float64)
    co = np.zeros((num_meds, num_meds), dtype=np.float64)
    for patient in records:
        for visit in patient:
            meds = sorted({m for m in visit[2] if 0 <= m < num_meds})
            for m in meds:
                counts[m] += 1
            for i, j in combinations(meds, 2):
                co[i, j] += 1
                co[j, i] += 1
    return counts, co


def row_normalize(matrix):
    sums = matrix.sum(axis=1, keepdims=True)
    return matrix / (sums + EPS)


def build_graph_arrays(records, num_meds):
    counts, co = cooccurrence_counts(records, num_meds)
    directed = co / (counts[:, None] + EPS)
    np.fill_diagonal(directed, 0.0)
    binary = (co > 0).astype(np.float64)
    np.fill_diagonal(binary, 0.0)
    return {
        "a_out": row_normalize(directed).astype(np.float32),
        "a_in": row_normalize(directed.T).astype(np.float32),
        "b_out": row_normalize(binary).astype(np.float32),
        "b_in": row_normalize(binary.T).astype(np.float32),
    }
