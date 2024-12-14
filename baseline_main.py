from model_words_baseline import *
import pandas as pd
from itertools import combinations
import random
from msm import *
from min_hash import *
from model_words import *
from LSH import *
import matplotlib.pyplot as plt
from evaluation import *

df_baseline = json_to_df_baseline("TVs-all-merged-cleaned-baseline.json")

bs_res = bootstrap_lsh(df_baseline, 1600, 0.7, 1)
print(bs_res)