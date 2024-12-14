import pandas as pd
from itertools import combinations
import random
from msm import *
from min_hash import *
from model_words import *
from LSH import *
import matplotlib.pyplot as plt




def check_candidate_pairs(index_tuple, df):
    a, b = index_tuple
    model_id_a, model_id_b = df.loc[a, "model_id"], df.loc[b, "model_id"]
    return model_id_a == model_id_b


def check_LSH_pairs(candidate_pairs, df):
    total_pairs = len(candidate_pairs)
    correct_pairs = 0
    for pair in candidate_pairs:
        if check_candidate_pairs(pair, df):
            correct_pairs += 1
    
    print(f"Total candidate pairs LSH = {total_pairs}")
    print(f"Total correct pairs LSH = {correct_pairs}")
    print(f"Fraction of correct pairs LSH= {correct_pairs/total_pairs}")
    return correct_pairs

def check_MSM_pairs(candidate_pairs, df, string):
    total_pairs = len(candidate_pairs)
    correct_pairs = 0
    for pair in candidate_pairs:
        if check_candidate_pairs(pair, df):
            correct_pairs += 1
    
    print(f"Total candidate pairs MSH {string} = {total_pairs}")
    print(f"Total correct pairs MSM {string} = {correct_pairs}")
    print(f"Fraction of correct pairs MSM {string} = {correct_pairs/total_pairs}")
    return correct_pairs

def get_all_true_pairs(df):
    n = df.shape[0]
    result_dict = dict()
    for i in range(n):
        model_id = df.loc[i, 'model_id']
        if model_id not in result_dict:
            result_dict[model_id] = []
        result_dict[model_id].append(i)
    
    filtered_dict = {k: v for k, v in result_dict.items() if len(v) > 1}

    all_pairs = set()
    for values in filtered_dict.values():
        if len(values) > 2:
            all_pairs.add(combinations(tuple(values), 2))
        elif len(values) == 2:
            all_pairs.add(tuple(values))
    print(f"# of true pairs {len(all_pairs)}")
    return all_pairs

def f1_score(candidate_pairs, true_pairs):
    TP = len(true_pairs.intersection(candidate_pairs))
    FP = len(candidate_pairs) - TP
    FN = len(true_pairs) - TP

    # recall
    if (TP + FN) > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    # precision
    if (TP + FP) > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    if (recall + precision) > 0:
        f1_score = (2*recall*precision) / (recall + precision)
    else:
        f1_score = 0
    
    return f1_score


def f1_score_msm(candidate_pairs_LSH, true_pairs, candidate_pairs_MSM):
    # possible true pairs = 
    print(f"len candidate_pairs_LSH = {len(candidate_pairs_LSH)}")
    print(f"len true pairs from test/ train = {len(true_pairs)}")
    print(f"len predicted pairs of MSM = {len(candidate_pairs_MSM)}")

    true_pairs_from_LSH = candidate_pairs_LSH.intersection(true_pairs)
    TP = len(true_pairs_from_LSH.intersection(candidate_pairs_MSM))

    # TP = len(true_pairs.intersection(candidate_pairs_MSM))
    FP = len(candidate_pairs_MSM) - TP
    FN = len(true_pairs_from_LSH) - TP

    # recall
    if (TP + FN) > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    # precision
    if (TP + FP) > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    if (recall + precision) > 0:
        f1_score = (2*recall*precision) / (recall + precision)
    else:
        f1_score = 0
    
    return f1_score

def pair_qualitiy(candidate_pairs, true_pairs):
    if len(candidate_pairs) > 0:
        return len(true_pairs.intersection(candidate_pairs)) / len(candidate_pairs)
    else:
        return 0

def pair_completeness(candidate_pairs, true_pairs):
    return len(true_pairs.intersection(candidate_pairs)) / len(true_pairs)

def F1_star(pair_qualitiy, pair_completeness):
    if pair_qualitiy > 0 or pair_completeness > 0:
        return (2*pair_qualitiy*pair_completeness) / (pair_qualitiy + pair_completeness)
    else:
        return 0


                                
def create_test_set_old(df, split_ratio, seed):
    random.seed(seed)
    n_products = df.shape[0]
    n_products_train = round(n_products*split_ratio)


    index_train = random.sample(set_index_full, n_products_train)

    # The test set will be the remaining indices
    index_test = list(set_index_full - set(index_train))

    # Create the training and test sets
    df_train = df.iloc[index_train]
    df_test = df.iloc[index_test]

    return df_train, df_test

def create_test_set(df, split_ratio, seed):
    random.seed(seed+2)
    n_products = df.shape[0]
    n_products_train = round(n_products * split_ratio)

    # Select a random starting point
    start_index = random.randint(0, n_products - 1)

    # Calculate the end index
    end_index = (start_index + n_products_train) % n_products

    # Handle the wrap-around case
    if end_index <= start_index:
        index_train = list(range(start_index, n_products)) + list(range(0, end_index))
    else:
        index_train = list(range(start_index, end_index))

    # The test set will be the remaining indices
    set_index_full = set(range(n_products))
    index_test = list(set_index_full - set(index_train))

    # Create the training and test sets
    df_train = df.iloc[index_train]
    df_test = df.iloc[index_test]

    return df_train, df_test

def create_test_set_replacement(df, split_ratio, seed):
    random.seed(seed)
    n_products = df.shape[0]

    # Sample indices with replacement for the training set
    train_indices = [random.randint(0, n_products - 1) for _ in range(round(n_products * split_ratio))]

    # Generate the test set indices (remaining rows not included in training set)
    # Since we are sampling with replacement, some rows may not be in the training set
    set_index_full = set(range(n_products))
    train_indices_set = set(train_indices)
    test_indices = list(set_index_full - train_indices_set)

    # Create the training and test DataFrames
    df_train = df.iloc[train_indices]
    df_test = df.iloc[test_indices]

    return df_train, df_test


def get_bands(k):
    bands = [i for i in range(5, k) if k % i == 0]
    r = [k / b for b in bands]
    t = []
    for i in range(len(bands)):
        t.append((1 / bands[i]) ** (1 / r[i]))
    
    new_b = []
    new_r = []
    new_t = []
    for i in range(len(bands)):
        # we want a t between 0.8
        if t[i] <= 0.94 and t[i] >= 0.20:
            new_b.append(bands[i])
            new_r.append(r[i])
            new_t.append(t[i])

        
    return new_b, new_r, new_t

# bands1 = get_bands(1600)[0]
# print(f" the bands are: {bands1}")


def bootstrap_lsh(df,k, split_ratio, n_bootstraps):

    
    
    mean_result = []
    for n in range(n_bootstraps):
        print(f"This is bootstrap number {n+1}/{n_bootstraps}")

        # df_train_model_id, df_train_info, df_test_model_id, df_test_info = create_test_set(df, 0.70, seed=n)

        df_train, df_test = create_test_set(df,split_ratio= 0.70, seed=n)


        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
   

        true_pairs_train = get_all_true_pairs(df_train)
        true_pairs_test = get_all_true_pairs(df_test)

        P = df_train.shape[0]
        n_all_possible_pairs = P*(P-1)/2
        
        bands, rows, t_values = get_bands(k)
        binary_matrix_train = create_binary_vector(df_train)
        signature_matrix = min_hash(binary_matrix_train, k)

        result_lsh = []
        
        for i in range(len(bands)):
            print(i)
            b = bands[i]
            r = rows[i]
            t = t_values[i]

            
            candidate_pairs = LSH(signature_matrix, b)
            n_found_pairs = check_LSH_pairs(candidate_pairs, df_train)

            PC = pair_completeness(candidate_pairs, true_pairs_train)
            PQ  = pair_qualitiy(candidate_pairs, true_pairs_train)
            f1_str = F1_star(PQ, PC)

            fract_comparisons = len(candidate_pairs)/n_all_possible_pairs

            result_lsh.append(np.array([k,b,r,t, len(candidate_pairs), n_found_pairs, PC, PQ, f1_str, fract_comparisons]))
        
        mean_result.append(result_lsh)
    
    return mean_result

df = json_to_df("TVs-all-merged-cleaned-brands.json")

# bs_res = bootstrap_lsh(df, 1600, 0.7, 1)
# print(bs_res)


# print(bs_res)


def bootstrap_msm(df,k, split_ratio, n_bootstraps):
    distance_tresholds = np.arange(0.05, 1, 0.05).tolist()


    # check references
    alpha=0.602
    beta=0
    gamma = 0.756
    eps = 0.522
    mu = 0.65
    delta = 0.7


    mean_result_bootstrap_msm = []
    for n in range(n_bootstraps):
        print(f"This is bootstrap number {n+1}/{n_bootstraps}")
        result_bootstrap = []

        # df_train, df_test = create_test_set(df, split_ratio, seed=n)
        df_train, df_test = create_test_set_replacement(df, split_ratio, seed=n)

        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
   

        true_pairs_train = get_all_true_pairs(df_train)
        true_pairs_test = get_all_true_pairs(df_test)

        P = df_train.shape[0]

        bands, rows, t_values = get_bands(k)
        binary_matrix_train = create_binary_vector(df_train)
        signature_matrix_train = min_hash(binary_matrix_train, k)

        
        for i in range(len(bands)):
        # for i in range(1):
            results_band = []

            # TRAIN 
            b = bands[i]
            r = rows[i]
            t = t_values[i]

            # signature_matrix_train = min_hash(binary_matrix_train, k)
            
            candidate_pairs_train = LSH(signature_matrix_train, b)
            n_found_pairs_train = check_MSM_pairs(candidate_pairs_train, df_train, string="train")
            # found_true_pairs_train = get_all_true_pairs()	

            MSM_sim_matrix_train = MSM(df, candidate_pairs_train, alpha, beta, delta, gamma, eps, mu)

            best_f1_train = 0
            best_dist_tresh = 0
            for dis in distance_tresholds:
                
                predicted_pairs_train = clustering(MSM_sim_matrix_train, dis)
                # check of dit klopt
                f1_1_train = f1_score_msm(candidate_pairs_train, true_pairs_train, predicted_pairs_train)
                print(f"for band {b},distance_threshold {dis} the f1_1_train score = {f1_1_train}")
                if f1_1_train > best_f1_train: 
                    best_f1_train = f1_1_train
                    best_dist_tresh = dis
            print(f"best distance threshold = {best_dist_tresh}")
                    
            # TEST
            binary_matrix_test = create_binary_vector(df_test)
            signature_matrix_test = min_hash(binary_matrix_test, k)
            
            candidate_pairs_test = LSH(signature_matrix_test, b)
            n_found_pairs_test = check_MSM_pairs(candidate_pairs_test, df_test, string="test")

            MSM_sim_matrix_test = MSM(df, candidate_pairs_test, alpha, beta, delta, gamma, eps, mu)
            

            # cluster onn the best found distance treshold
            predicted_pairs_test = clustering(MSM_sim_matrix_test, best_dist_tresh)

            # lsh_corrected_pairs = candidate_pairs_test.intersection(true_pairs_test)
            # f_1_score_test = f1_score(predicted_pairs_test, lsh_corrected_pairs)

            # f_1_score_test = f1_score_msm(predicted_pairs_test, true_pairs_test)
            f_1_score_test = f1_score_msm(candidate_pairs_test, true_pairs_test, predicted_pairs_test)
            print(f"f1_score_msm = {f_1_score_test} after accounting for possible pairs out of lSH")

            

            # calculate length test 
            len_test = df_test.shape[0]
            test_total_comparisons = len_test*(len_test -1)/2
            test_fract_comparison = len(candidate_pairs_test) / test_total_comparisons

            result_bootstrap.append([b,r,t, len(candidate_pairs_test), n_found_pairs_test, f_1_score_test, test_fract_comparison])
        mean_result_bootstrap_msm.append(result_bootstrap)

    return mean_result_bootstrap_msm




                





                        
                        









                    
                    

                    







            
            
