from primer_manager import PrimerManager
from gpt2_for_reflections import GPT2ForReflections
from rqc import ReflectionQualityClassifier

from helper_functions import *
from generate_reflections_over_dataset import generate_reflections_over_dataset

import pandas as pd
from tqdm import tqdm
import itertools

# setting global seed for reproducability
INIT_SEED = 100
set_global_seeds(INIT_SEED)

def dict_of_lists_to_list_of_dicts(dict_of_list):
    # make sure every entry is a list
    dict_of_list = {key: val if type(val) == list else [val] for key, val in dict_of_list.items()}

    # this confusing code is basically taking the cartensian product of every list in the dictionary
    keys = dict_of_list.keys()
    combinations = itertools.product(*dict_of_list.values())
    list_of_dicts = [ dict(zip(keys, comb)) for comb in combinations ]

    return list_of_dicts


def grid_search(data_set, Primers, GPT2FR, hyperparameters_dict, debug=False, save_dir="."):

    # the logging and saving doesn't really work as well if there are multiple search types
    if "search_type" in hyperparameters_dict:
        if type(hyperparameters_dict["search_type"]) == list:
            if len(hyperparameters_dict["search_type"]) > 1:
                raise ValueError("Please only select 1 value for hyperparameter 'search_type'")

    # convert dictionary of lists to list of dictionaries
    hyperparameters_list = dict_of_lists_to_list_of_dicts(hyperparameters_dict)

    df = None
    for hyperparameters in hyperparameters_list:
        # running test
        output_df = generate_reflections_over_dataset(data_set, Primers, GPT2FR, hyperparameters, permutations=3, primer_types=["random"], debug=debug, save_dir=SAVE_DIR)

        # saving output
        if df is None:
            df = output_df
        else:
            df = df.append(output_df, ignore_index=True)
    
    return df

if __name__ == "__main__":
    # parse commandline (helper_functions.py contains code)
    args = get_args()

    hyperparameters = {
        "num_shots": 5,
        "repetition_penalty": 1.0,
        "top_k": [0, 10, 100],
        "temperature": [0.01, 0.1, 0.5, 1.0],
        "seed": INIT_SEED,
        "search_type": "sample"
    }

    print("\nLoading Primers...")
    primer_df = pd.read_csv('static_data/CAMH Primers/CAMH_primers_categorized.csv', index_col=0)
    Primers = PrimerManager(primer_df, seed=hyperparameters["seed"])

    # The dataset and primer set will be the same here
    df = pd.read_csv('static_data/CAMH Primers/CAMH_primers_categorized.csv', index_col=0)

    print("\nLoading model...")
    GPT2FR = GPT2ForReflections(model_name=args.model)

    
    print("\nRunning hyperparameter search...")
    SAVE_DIR = "generated_data"
    df = grid_search(df, Primers, GPT2FR, hyperparameters, debug=args.debug, save_dir=SAVE_DIR)

    print("\nSaving to csv...")
    df.to_csv(f"{SAVE_DIR}/grid_search.csv")
