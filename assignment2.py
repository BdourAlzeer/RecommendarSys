import pandas as pd
import numpy as np
from itertools import combinations, chain
import matplotlib.pyplot as plt
import time
import os

import pickle

def convert_signature_in_bands(signature_matrix, band_size):
    bands_split = list(range(0, signature_matrix.shape[0], band_size))
    bands = []
    for i in range(0, len(bands_split)):
        try:
            bands.append(signature_matrix[bands_split[i]: bands_split[i+1]])
        except IndexError:
            bands.append(signature_matrix[bands_split[i]:])
    return bands


def hash_column(col, bucket_size=10000):
    return hash(tuple(col)) % bucket_size


def get_signature_matrix(p, data_frame):
    signature_matrix = pd.DataFrame()  # creates a new dataframe that's empty

    def get_signature(col):
        return col.index[col == 1].tolist()[0]

    for i in range(0, p):
        np.random.seed(i)
        shuffle_df = data_frame.iloc[np.random.permutation(data_frame.shape[1])]
        sig_vector = shuffle_df.apply(get_signature, axis=0)
        signature_matrix = signature_matrix.append(sig_vector, ignore_index=True)
    return signature_matrix


def map_band_columns_into_bucket(bucket_size, sig_band):
    buckets = {}

    def map_col_to_bucket(col):
        col_index = col.name
        hash_key = hash_column(col, bucket_size)
        if hash_key not in buckets:
            buckets[hash_key] = []
        buckets[hash_key].append(col_index)

    sig_band.apply(map_col_to_bucket, axis=0)
    return buckets


def calculate_triangular_matrix(p_pair):
    i = sorted(p_pair)
    return str(i)
    # k = i-1
    # return k*(total_pairs_count - (i/2))+j-i


def signature_jaccard_similarity_of_columns(signature_matrix, candidate_pair):
    col_1, col_2 = candidate_pair
    s1 = signature_matrix[col_1] - signature_matrix[col_2]
    total = signature_matrix.shape[0]
    sum_0 = total - s1.astype(bool).sum(axis=0)
    return sum_0 / float(total)


#TODO Question 7
def get_jaccard_similarity_between_two_columns(data_matrix, pair):
    col1, col2 = pair
    intersection = np.sum(data_matrix[col1] & data_matrix[col2])
    union = np.sum(data_matrix[col1] | data_matrix[col2])
    return intersection/float(union)


def get_false_positive_signature(all_candidate_pairs, sig_pair_count_matrix, total_pairs):
    print('Calculating False Positive Signature')
    false_positive_pairs = []
    false_positive = 0
    for c_pair in all_candidate_pairs:
        k = calculate_triangular_matrix(c_pair)
        try:
            sig_pair_count_matrix[k]
        except KeyError:
            false_positive_pairs.append(c_pair)
            false_positive += 1
    return false_positive, false_positive_pairs


def get_false_negative_signature(all_pairs_possible, candidate_pairs, sig_pair_count_matrix):
    print('Calculating False Negative Signature')
    false_negative = 0
    for c_pair in all_pairs_possible:
        k = calculate_triangular_matrix(c_pair)
        try:
            sig_pair_count_matrix[k]
            if c_pair not in candidate_pairs:
                false_negative += 1
        except KeyError:
            pass
    return false_negative


# TODO: Probability of a candidate pair
def calculate_probability_of_candidate_pair(s, b, r):
    return 1 - np.power((1 - np.power(s, r)), b)


def get_graph_data(permutations, possible_band_sizes, t=0.3):
    possible_s = np.arange(0.0, 1.0, 0.1)
    probability_on_possible_band_size = list()
    best_delta = 0.00
    best_band_size = None
    for band_size in possible_band_sizes:
        bands = permutations / band_size
        current = {
            'band_size': bands,
            'row_size': band_size,
            'prob_candidate_pair': [],
            's_values': possible_s
        }

        for s in possible_s:
            current['prob_candidate_pair'].append(
                calculate_probability_of_candidate_pair(s, bands, band_size))
            if np.isclose(s, t):
                delta = current['prob_candidate_pair'][int(t * 10)] - current['prob_candidate_pair'][
                    int(t * 10) - 1]
                current['delta'] = delta

        if current['delta'] > best_delta:
            best_delta = delta
            best_band_size = band_size

        probability_on_possible_band_size.append(current)
    return probability_on_possible_band_size, best_band_size, permutations/best_band_size


def draw_s_curve(graph_data, p, threshold=0.3):
    s_curve_figure = plt.figure(figsize=(100, 80))
    s_curve_figure.suptitle('S Curve for all possible number of bands (b) and rows in bands (r) for Permutation %s' % p)

    for idx, graph in enumerate(graph_data):
        curve = s_curve_figure.add_subplot(2, np.ceil(len(graph_data)/2), idx + 1)
        curve.set_title('Bands (b) : %s | Rows(r): %s' % (graph['band_size'], graph['row_size']))
        curve.plot([threshold, threshold], [0, 1], color='g')
        curve.plot(graph['s_values'], graph['prob_candidate_pair'], color='b')

        delta_text = 'delta(s=%.1f->%.1f): %.4f' % (graph['s_values'][int(threshold*10)-1],
                                             graph['s_values'][int(threshold*10)], graph['delta'])
        plt.text(0.6, 0.2, delta_text, fontsize=9, ha='center', va='center', transform=curve.transAxes)

    plt.show()


# TODO: Question 6
def remove_false_positives_from_signature_matrix(all_candidate_pairs, false_positive_pairs):
    cand_pairs_no_fp_pairs = set(all_candidate_pairs) - set(false_positive_pairs)
    return cand_pairs_no_fp_pairs


def get_false_positive_from_original_data(all_pair_count_matrix, similar_pairs_without_fp, t=0.3):
    false_positive = 0
    for positive_pairs in similar_pairs_without_fp:
        k = calculate_triangular_matrix(positive_pairs)
        try:
            all_pair_count_matrix[k]
        except KeyError:
            false_positive += 1
    return false_positive


def get_false_negative_from_original_data(all_pair_count_matrix, all_pairs, similar_pairs_without_fp):
    false_negative = 0
    for pair in all_pairs:
        k = calculate_triangular_matrix(pair)
        try:
            all_pair_count_matrix[k]
            if pair not in similar_pairs_without_fp:
                 false_negative += 1
        except KeyError:
            pass
    return false_negative


def draw_fp_fn_graph(fp_data, fn_data):
    fp_data = {int(key): fp_data[key] for key in fp_data}
    fn_data = {int(key): fp_data[key] for key in fn_data}
    fp_data = sorted(fp_data.items())
    fn_data = sorted(fn_data.items())
    plt.figure(10)
    plt.rcParams.update({'font.size': 15})
    plt.plot([i[0] for i in fp_data], [i[1] for i in fp_data], label='False Positive')
    plt.plot([i[0] for i in fn_data], [i[1] for i in fn_data], label='False Negative')
    plt.ylabel('False Positive and Negatives')
    plt.title('False Positive and Negative On different No of Bands')
    plt.xlabel('Bands')
    plt.legend()
    plt.show()


def get_candidate_pair_from_buckets(signature_bands):
    candidate_pairs = []
    for i in range(0, len(signature_bands)):
        buckets = map_band_columns_into_bucket(10000, signature_bands[i])
        for h in buckets:
            if len(buckets[h]) >= 2:
                candidate_pairs = candidate_pairs + [comb for comb in combinations(buckets[h], 2)]
    return candidate_pairs


def lsh_give_b_r(all_pairs, signature, sig_pair_count_matrix, all_pair_count_matrix, r, p):
    signature_bands = convert_signature_in_bands(signature, r)
    total_pairs = len(all_pairs)
    candidate_pairs = list(set(get_candidate_pair_from_buckets(signature_bands)))
    fp, false_positive_pairs = get_false_positive_signature(candidate_pairs, sig_pair_count_matrix, total_pairs)
    fn = get_false_negative_signature(all_pairs, candidate_pairs, sig_pair_count_matrix)
    s = len(candidate_pairs)
    similar_pairs_without_fp = remove_false_positives_from_signature_matrix(candidate_pairs, false_positive_pairs)
    print('b: %s, Candidate Pairs: %s, False Positive,False Negatives: %s,%s, True Positives: %s'
          % (int(p/r), s, fp, fn, len(similar_pairs_without_fp)))
    fp_all_positive = get_false_positive_from_original_data(all_pair_count_matrix, similar_pairs_without_fp)
    fn_all_positive = get_false_negative_from_original_data(all_pair_count_matrix, all_pairs, similar_pairs_without_fp)

    print('False Positive, False Negative: %s,%s' % (fp_all_positive, fn_all_positive))

    return fp, fn, similar_pairs_without_fp, fp_all_positive, fn_all_positive


def count_pair_all_data(df, all_pairs, t=0.3):
    original_pair_matrix = dict()
    for pair in all_pairs:
        sim = get_jaccard_similarity_between_two_columns(df, pair)
        if sim >= t:
            k = calculate_triangular_matrix(pair)
            original_pair_matrix[k] = 1
    return original_pair_matrix


def count_pair_signature_data(signature_matrix, all_pairs, t=0.3):
    sig_pair_matrix = dict()
    for pair in all_pairs:
        sim = signature_jaccard_similarity_of_columns(signature_matrix, pair)
        if sim >= t:
            k = calculate_triangular_matrix(pair)
            sig_pair_matrix[k] = 1
    return sig_pair_matrix


def assignment2(df, permutations, all_pairs, all_pair_count_matrix):
    all_fp = dict()
    all_fn = dict()
    all_b_fp = dict()
    all_b_fn = dict()
    all_tps = dict()
    possible_band_sizes = [d for d in range(2, permutations) if permutations % d == 0]
    graph_data, r, b = get_graph_data(permutations, possible_band_sizes)
    print('Best b: %s, r: %s for permutation %s' % (b, r, permutations))
    start_time = time.time()
    signature = get_signature_matrix(permutations, df)
    print('--- %s seconds --- Signature Matrix Created for Permutation: %s' % ((time.time() - start_time), permutations))
    start_time = time.time()
    sig_pair_count_matrix = count_pair_signature_data(signature, all_pairs)
    print('--- %s seconds --- Signature Pair Matrix Created for Permutation: %s'
          % ((time.time() - start_time), permutations))

    fp, fn, all_tp, tp_fp, tp_fn = lsh_give_b_r(all_pairs, signature, sig_pair_count_matrix, all_pair_count_matrix,
                                                r, permutations)
    all_fp[b] = fp
    all_fn[b] = fn
    all_b_fp[b] = tp_fp
    all_b_fn[b] = tp_fn
    all_tps[b] = all_tp

    for no_row in possible_band_sizes:
        band = permutations/no_row
        if band not in all_b_fp:
            print('Performing for b: %s, r: %s for permutation %s' % (band, no_row, permutations))
            fp, fn, all_tp, tp_fp, tp_fn = lsh_give_b_r(all_pairs, signature, sig_pair_count_matrix, all_pair_count_matrix,
                                                        no_row, permutations)
            all_fp[band] = fp
            all_fn[band] = fn
            all_b_fp[band] = tp_fp
            all_b_fn[band] = tp_fn
            all_tps[b] = all_tp

    result = dict()
    result['all_tp'] = all_tps
    result['fp_6'] = all_fp
    result['fn_6'] = all_fn
    result['fp_7'] = all_b_fp
    result['fn_7'] = all_b_fn
    result['graph'] = graph_data
    return result


# return all_fp, all_fn, all_b_fp, all_b_fn, graph_data
start_time = time.time()
data = pd.read_csv('data-Assignment2.csv', header=None)
all_pairs = [tuple(sorted(comb)) for comb in combinations(data.columns.values, 2)]
all_pair_count_matrix = count_pair_all_data(data, all_pairs)

print("--- All Pair count Matrix %s seconds ---" % (time.time() - start_time))
start_time = time.time()
result_100 = assignment2(data, 100, all_pairs, all_pair_count_matrix)
result_500 = assignment2(data, 500, all_pairs, all_pair_count_matrix)
print("--- %s seconds ---" % (time.time() - start_time))

draw_fp_fn_graph(result_100['fp_6'], result_100['fn_6'])
draw_fp_fn_graph(result_100['fp_6'], result_100['fn_6'])
draw_fp_fn_graph(result_500['fp_7'], result_100['fn_7'])
draw_fp_fn_graph(result_500['fp_7'], result_100['fn_7'])













