# Part 1: Matching Key-Valye pairs.
#   - q-grams as similarity measure / cosine / winkler can be used

# Part 2: HSM Key-value pairs that were not matched in Part 1
#   - use model worsd to calc percentages of matches
#   - uses additional "learned" weight to find simalirity between products
#   - uses paper HSM: https://personal.eur.nl/frasincar/papers/CAISE2013/caise2013.pdf, A Hybrid Model Words-Driven Approach
#     for Web Product Duplicate Detection

# Part 3: TMWM model words from title
#   - uses paper: https://repub.eur.nl/pub/62361/1-s2.0-S0167923612000681-main.pdf, Faceted product search powered by the Semantic Web

## MSM from file:///C:/Users/timod/Desktop/Msc%20Econometrics%20BAQM/2_CS/assignment/Multi-component%20similarity%20%20method%20for%20Web%20product%20duplicate%20detection..pdf

from strsimpy.qgram import QGram
from model_words import *
from itertools import combinations



def qgram(q, string):
    # generate q-gram for string
    return set([string[i:i+q] for i in range(len(string) - q + 1)])

def calcSim_q(s1, s2, q):
    qgram = QGram(q)

    n1 = len(s1)
    n2 = len(s2)

    qgram_dist = qgram.distance(s1, s2)

    return (n1 + n2 - qgram_dist)/(n1+n2)


def diff_brands(index_tuple, df):
    a, b = index_tuple

    # since in the data cleaning we extended the model 
    if "brand" in df.loc[a, "featuresMap"] and "brand" in df.loc[b, "featuresMap"]:
        # return true if they are different 
        return df.loc[a, "featuresMap"]["brand"] != df.loc[b, "featuresMap"]["brand"]
    # for just a few we dont know the brand name (14) so if brand isnt in feature map, still investigate -> return false
    return False

def same_shop(index_tuple, df):
    a, b = index_tuple
    return df.loc[a, "shop"] == df.loc[b, "shop"]

def exMW(p, keys, df):
    result = set()
    for key in list(keys):
        value = df.loc[p, "featuresMap"][key]
        result.update(model_words_values(value))
    return result

def exMW_title(p,df):
    return df.loc[p, "model_words_title"]

def mw(C,D):
    len_union = len(C.union(D))
    return len(C.intersection(D)) / len_union if len_union > 0 else 0

def minFeatures(i, j, df):
    return min(len(df.loc[i, "featuresMap"]), len(df.loc[j, "featuresMap"]))


# TWMW
def get_product_title(i,j,df):
    return df.loc[i, "title"], df.loc[j, "title"]

from strsimpy.levenshtein import Levenshtein
def lv(x,y):
    lev = Levenshtein()
    distance = abs(lev.distance(x,y))
    denom = max(len(x), len(y))

    return distance/ denom if denom != 0 else 0

def calcCosine(x,y):
    a = set(x.split())
    b = set(y.split())
    denom = ((len(a))**0.5 * (len(b))**0.5)
    return len(a.intersection(b)) / denom if denom != 0 else 0

# a = calcCosine("sony ericsson X1", "xperia X1")
# print(a)

def avgLvSim(x,y):
    tot = 0
    total_length = 0
    for i in x:
        for j in y:
            tot += ((1 - lv(i,j)) * (len(i) + len(j)))
            total_length += len(i) + len(j)

    # is this a distance or simalarity>
    return tot / total_length if total_length != 0 else 0


# a = avgLvSim(set(["D700", "12.1MP"]), set(["D700", "4MP"]))
# print(a)

def get_numeric_alphabetic(word):
    numeric_part = ''.join(filter(str.isdigit, word))
    nonnumeric_part = ''.join(filter(str.isalpha, word))
    return nonnumeric_part, numeric_part


def avgLvSimMW(x,y, threshold):
    # might have to extract brand from this fucntion! since brand has been added to MW title (only alphabetic):
    # mw_title_x, mw_title_y = df.loc[x,"model_words_title"], df.loc[y,"model_words_title"]
    # brand_x, brand_y = df.loc[x,"brand"], df.loc[y,"brand"]

    # mw_title_x = mw_title_x.remove(brand_x)
    # mw_title_y = mw_title_y.remove(brand_y)

    nominator = 0
    denominator = 0

    for word_x in x:
        alpha_x, numeric_x = get_numeric_alphabetic(word_x)
        # for word_y in list(mw_title_y):
        for word_y in y:
            alpha_y, numeric_y = get_numeric_alphabetic(word_y)
            if avgLvSim(alpha_x, alpha_y) > threshold and numeric_x == numeric_y:
                nominator += (1-lv(word_x, word_y))*(len(word_x) + len(word_y))
                denominator += len(word_x) + len(word_y)

    return nominator / denominator if denominator != 0 else 0

def TWMW(i, j, alpha, beta, delta, eps, df):

    a, b = get_product_title(i,j,df)
    # a, b = productName = title

    nameCosineSim = calcCosine(a,b)
    if nameCosineSim > alpha:
        return 1

    modelWordsA = list(exMW_title(i, df))
    modelWordsB = list(exMW_title(j, df))

    
    #  # compute final similarity
    # finalNameSim = beta * nameCosineSim + (1-beta) * avgLvSim(a,b)

    # check if products are different
    for wordA in modelWordsA:
        for wordB in modelWordsB:
            nonnumeric_a, numeric_a = get_numeric_alphabetic(wordA)
            nonnumeric_b, numeric_b = get_numeric_alphabetic(wordB)
            # if non-numeric approx the same and Numeric != same return false: -1 -> productname does not 
            # CHECK IF WE USE EPS FOR THE APPROX SIMILARITY
            if lv(nonnumeric_a, nonnumeric_b) < eps and numeric_a != numeric_b:
                return -1 
            # elif lv(nonnumeric_a, nonnumeric_b) > eps and numeric_a == numeric_b:
            #     similar = True
            #     modelWordsSimVal = avgLvSimMW(modelWordsA, modelWordsB, threshold=eps)
            #     break

    # compute final similarity
    finalNameSim = beta * nameCosineSim + (1-beta) * avgLvSim(a,b)

    # Check if we have a pair of modelwords that are likely to be the same   
    similar = False
    for wordA in modelWordsA:
        for wordB in modelWordsB:
            nonnumeric_a, numeric_a = get_numeric_alphabetic(wordA)
            nonnumeric_b, numeric_b = get_numeric_alphabetic(wordB)
            # if at least one of the pair non-num are approx the same AND num== same 
            if lv(nonnumeric_a, nonnumeric_b) > eps and numeric_a == numeric_b:
                similar = True
                break
    
    if similar:
        modelWordsSimVal = avgLvSimMW(modelWordsA, modelWordsB, threshold=eps)
        finalNameSim = delta* modelWordsSimVal + (1-delta)*finalNameSim
    
    return finalNameSim if finalNameSim > eps else -1

from sklearn.cluster import AgglomerativeClustering

def clustering(M_dist, dist_treshold):
    clustering = AgglomerativeClustering(n_clusters=None, linkage="single",
                                          distance_threshold= dist_treshold,
                                          metric="precomputed")
    
    cluster_matrix = clustering.fit_predict(M_dist)

    cluster_dict = dict()
    for prod, cluster in enumerate(cluster_matrix):
        # cluster_dict[cluster].append(prod)
        cluster_dict.setdefault(cluster, []).append(prod)
    
    duplicate_pairs = set()
    for cluster_members in cluster_dict.values():
        if len(cluster_members) >= 2:
            for pair in combinations(cluster_members,2):
                duplicate_pairs.add(pair)

    return duplicate_pairs

def MSM(df, candidate_pairs, alpha, beta, delta, gamma, eps, mu):
    # create matrix containing pairwise similarities
    n = df.shape[0]
    M_dist = np.full([n,n], 1e10)


    # i, j are product indexes
    for i,j in candidate_pairs:
        if not same_shop((i,j), df) and not diff_brands((i,j), df):
            sim = 0
            avgSim = 0
            m = 0 #number of matches
            w = 0 #weight of mathces

            keys_i = set(df.loc[i, "featuresMap"].keys())
            keys_j = set(df.loc[j, "featuresMap"].keys())

            nmk_i = keys_i.difference(keys_j) # non matching keys of i
            nmk_j = keys_j.difference(keys_i) # non matching keys of j

            for key_i, value_i in df.loc[i, "featuresMap"].items():
                for key_j, value_j in df.loc[j, "featuresMap"].items():
                    keySim = calcSim_q(key_i, key_j,q=3)
                    if keySim > gamma:
                        valueSim = calcSim_q(value_i, value_j,q=3)
                        weight = keySim
                        sim += weight*valueSim
                        m += 1
                        w += weight
                        nmk_i.discard(key_i)
                        nmk_j.discard(key_j)
            
            if w > 0:
                avgSim = sim/w
            
            mwPerc = mw(exMW(i, nmk_i, df), exMW(j, nmk_j, df))
            titleSim = TWMW(i,j,alpha, beta, delta, eps, df)

            if titleSim == -1:
                theta_1 = m/minFeatures(i,j,df)
                theta_2 = 1 - theta_1
                hSim = theta_1 * avgSim + theta_2 * mwPerc
            else:
                theta_1 = (1- mu)* m / minFeatures(i, j,df)
                theta_2 = 1- mu - theta_1
                hSim = theta_1* avgSim + theta_2*mwPerc + mu*titleSim
            
            M_dist[i,j] = 1 -hSim # transform to dissimilarity
    
    return M_dist




            



















            


    







