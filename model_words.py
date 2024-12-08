import json
import re
import numpy as np
import pandas as pd
from functools import reduce

    # # model words title
    # df["model_words_title"] = df["title"].apply(model_words_title)

    # # model words values
    # df["model_words_features"] = df["featuresMap"].apply(apply_model_words_featuresMap)



def model_words_title(title):
    results = []

    # numeric+alphabetic or other way around + only numeric if decimal . exists
    # pattern = r"([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)"
    # pattern = r"([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+)|([0-9]+\.[0-9]+))[a-zA-Z0-9]*)"
    pattern = r'([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^\d, ]+[0-9]+))[a-zA-Z0-9]*)'

    


    matches = re.findall(pattern, title)
    
    for match in matches: 
            results.append(match[0])


    return set(results)

def model_words_values(value):
    results = []

    pattern = r"(^\d+(\.\d+)?[a-zA-Z]*$|^\d+(\.\d+)?$)"
    matches = re.findall(pattern, value)
    
    for match in matches: 

        #extract numeric part removing, alphabetic characters
        numeric_part = re.match(r'^\d+(\.\d+)?', match[0])
        if numeric_part:
            results.append(numeric_part.group())
            
    return set(results)




def apply_model_words_featuresMap(feature_map):
    result = []
    including_features = [""]
    for key, value in feature_map.items():
        if key != "upc":
            result.append(model_words_values(value))
    
    if len(result) != 0:
        result_set = set([i for lst in result for i in lst])

    else:
        print("not working for ", feature_map)
        result_set = np.nan
    return result_set


def json_to_df(data):
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

    df["model_words_title"] = df["title"].apply(model_words_title)

    # Update the model_words_title column to include the brand for each row
    df["model_words_title"] = df.apply(lambda row: row["model_words_title"].union({row["brand"]}) if pd.notna(row["brand"]) else row["model_words_title"], axis=1)

    # excluded model_id out of matrix
    df["model_words_title"] = df.apply(lambda row: row["model_words_title"].difference({row["model_id"]}) , axis=1)





    # model words values
    df["model_words_features"] = df["featuresMap"].apply(apply_model_words_featuresMap)

    return df

df = json_to_df("TVs-all-merged-cleaned-brands.json")


def create_binary_vector(df):
    all_mw_title = list(reduce(set.union, df["model_words_title"], set()))
    all_mw_values = list(reduce(set.union, df["model_words_features"], set()))

    P = df.shape[0]
    # create empty signature matrix
    signature_matrix = np.zeros((len(all_mw_title), P))
    # Populate the signature matrix
    for i, word in enumerate(all_mw_title):
        for j in range(P):
            # Check if the word exists in df["model_words_title"][j]
            if (word in df["model_words_title"].iloc[j]) or (word in df["model_words_features"][j]):
                signature_matrix[i, j] = 1

    return signature_matrix



print(df.head(5))

mw = create_binary_vector(df)
print(mw.shape)


    
    