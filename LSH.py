from model_words import *
from itertools import combinations

def LSH(signature_matrix, b):

    n, n_products = signature_matrix.shape
    # n = b*r
    assert n % b == 0
    r = n // b

    buckets = {}
    for band_idx in range(b):
        start_row = band_idx * r
        end_row = start_row + r
        band = signature_matrix[start_row:end_row, :]

        for product_idx in range(n_products):
            band_signature = tuple(band[:, product_idx])
            if band_signature not in buckets:
                buckets[band_signature] = []
            buckets[band_signature].append(product_idx)
        


    # Generate candidate pairs from buckets
    candidate_pairs = set()
    for bucket in buckets.values():
        if len(bucket) > 1:
            for pair in combinations(bucket, 2):
                candidate_pairs.add(tuple(sorted(pair)))
    return candidate_pairs





