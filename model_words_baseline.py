import json
import re
import numpy as np
import pandas as pd
from functools import reduce

def model_words_title_baseline(title):
    results = []

    pattern = r'([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^\d, ]+[0-9]+))[a-zA-Z0-9]*)'
    matches = re.findall(pattern, title)
    
    for match in matches: 
            results.append(match[0])

    return set(results)


def model_words_values_baseline(value):
  feature_pattern = r'(^\d+(\.\d+)?[a-zA-Z]*$|^\d+(\.\d+)?$)'
                   
  regex = re.compile(feature_pattern)
  matches = regex.findall(value)

  # Extract and return the numeric part of each match
  model_words = []
  tokens = value.split()
  for token in tokens:
    matches = regex.findall(token)
    for match in matches:
        full_match = match[0]  # Full match from regex

        # only retain only the numeric part

        numeric_part = re.match(r'^\d+(\.\d+)?', full_match).group()
        model_words.append(numeric_part)

  return set(model_words)


def apply_model_words_featuresMap_baseline(feature_map):
    result = []
    including_features = [""]
    for key, value in feature_map.items():
        # if key != "upc":
        #     result.append(model_words_values(value))
        result.append(model_words_values_baseline(value))
    
    if len(result) != 0:
        result_set = set([i for lst in result for i in lst])

    else:
        print("not working for ", feature_map)
        result_set = np.nan
    return result_set


def json_to_df_baseline(data):
    with open(data, "r") as file:
        json_data = json.load(file)

    rows = []
    for model in json_data.values():
        for tv in model:
            brand = np.nan

            if "brand" in tv["featuresMap"].keys():
                brand = tv["featuresMap"]["brand"]
                
            new_row =  {"model_id": tv["modelid"],
                        "shop": tv["shop"],
                        "featuresMap": tv["featuresMap"],
                        "title": tv["title"],
                        "brand": brand}
            
            rows.append(new_row)

    df = pd.DataFrame(rows)

    all_model_ids = set(df["model_id"])

    df["model_words_title"] = df["title"].apply(model_words_title_baseline)


    # model words values
    df["model_words_features"] = df["featuresMap"].apply(apply_model_words_featuresMap_baseline)

    return df


def create_binary_vector_baseline(df):
    all_mw_title = reduce(set.union, df["model_words_title"], set())
    all_mw_values = reduce(set.union, df["model_words_features"], set())

    # rows = all 
    P = df.shape[0]

    union_title_values = list(all_mw_title.union(all_mw_values))
   
    union_title_values.sort()

    # create empty signature matrix

    binary_matrix = np.zeros((len(union_title_values), P))


    # for i, word in enumerate(all_mw_title):
    for i, word in enumerate(union_title_values):
        for j in range(P):
            # Check if the word exists in df["model_words_title"][j]
            if (word in df["model_words_title"].iloc[j]) or (word in df["model_words_features"][j]):
                binary_matrix[i, j] = 1

    print(f"Binary matrix shape = {binary_matrix.shape}")
    
    return binary_matrix





    
    