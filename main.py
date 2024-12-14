import pandas as pd
from itertools import combinations
import random
from msm import *
from min_hash import *
from model_words import *
from LSH import *
import matplotlib.pyplot as plt
from evaluation import *
from model_words_baseline import *

def plot(x_values, y_values, x_name, y_name, x_values_b, y_values_b):
    plt.figure(figsize=(8, 6))
    
    # First line
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label='MSPMB')
    
    # Second line
    plt.plot(x_values_b, y_values_b, marker='s', linestyle='--', color='r', label='MSPM+')
    
    # Labels and title
    plt.xlabel(f'{x_name}')
    plt.ylabel(f"{y_name}")
    plt.title(f'{y_name} vs {x_name} Plot')
    
    # Grid and legend
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig(f'{y_name} vs {x_name} Plot.png')
    
    # Show the plot
    plt.show()

    return

                
def convert_data_lsh_df(data, method):
    mean_data = [
    np.mean([run[i] for run in data], axis=0)  # Compute the mean for each index across all runs
    for i in range(len(data[0])) ]                      

    mean_data_10x10 = [list(arr[:10]) for arr in mean_data]

    # Output the result
    print(mean_data_10x10)
    df = pd.DataFrame(mean_data_10x10, columns=[
        "k", "b", "r", "t", "len(candidate_pairs)", 
        "n_found_pairs", "PC", "PQ", "f1_star", "fract_comparisons"
    ])
    df.to_excel(f"results_{method}.xlsx")
    return df


def convert_data_msm_df(data, method):
    mean_data = [
        np.mean([run[i] for run in data], axis=0)  # Compute the mean for each index across all runs
        for i in range(len(data[0]))              # Loop through the number of arrays in a single run
    ]

    # Convert the result into a 10x7 list by slicing only the first 7 elements of each array
    mean_data_10x10 = [list(arr[:7]) for arr in mean_data]

    # Output the result
    print(mean_data_10x10)
    df = pd.DataFrame(mean_data_10x10, columns=[
        "b", "r", "t", "len(candidate_pairs_test)", 
        "n_found_pairs_test", "f_1_score_test", "test_fract_comparison"
    ])

    # Display the dataframe
    df.to_excel(f"results_MSM_{method}.xlsx", index = False)



df_mspmb = json_to_df("TVs-all-merged-cleaned-brands.json")
df_baseline = json_to_df_baseline("TVs-all-merged-cleaned-baseline.json")

bs_lsh_baseline = bootstrap_lsh(df_baseline, 1600, 0.7, 5)
bs_lsh_mspmb = bootstrap_lsh(df_mspmb, 1600, 0.7, 5)

df_results_lsh_mspmb = convert_data_lsh_df(bs_lsh_mspmb, method="msmpb")
df_results_lsh_baseline = convert_data_lsh_df(bs_lsh_baseline, method="msmp+")
# print("Columns in df_baseline:", df_results_lsh_baseline.columns)

# bs_msm_baseline = bootstrap_msm(df_baseline, 1600, 0.7, 1)
# # bs_msm_mspmb = bootstrap_msm(df_mspmb, 1600, 0.7, 1)

# # df_results_msm_mspmb = convert_data_msm_df(bs_msm_mspmb, method="msmpb")
# df_results_msm_baseline = convert_data_msm_df(bs_msm_baseline, method="msmp+")



