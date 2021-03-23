from primer_manager import PrimerManager
from gpt2_for_reflections import GPT2ForReflections
from rqc import ReflectionQualityClassifier

from helper_functions import *
from k_fold import k_fold

import numpy as np
import pandas as pd
import torch
import random

from tqdm import tqdm
import itertools
import ast

import argparse
import traceback

# setting global seed for reproducability
INIT_SEED = 100
random.seed(INIT_SEED)
np.random.seed(INIT_SEED)
if not INIT_SEED is None:
    torch.manual_seed(INIT_SEED)


if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Testing Simple Reflection Generation')
    parser.add_argument('--model', type=str, default='gpt2', help="Model name")
    parser.add_argument('--debug', action="store_true", default=False)
    args = parser.parse_args()

    hyperparameters = {
        "num_shots": 3,
        "repetition_penalty": 1.0,
        "top_k": 50,
        "temperature": 0.5,
        "seed": INIT_SEED,
        "search_type": "sample"
    }

    print("\nLoading model...")
    GPT2FR = GPT2ForReflections(model_name=args.model)

    print("\nLoading Primers...")
    primer_file = "simple_reflections_gpt2.csv"
    #primer_file = "complex_reflections_gpt2.csv"
    #primer_file = "simple_reflections_human.csv"
    #primer_file = "complex_reflections_human.csv"
    primer_df = pd.read_csv(f'static_data/{primer_file}', index_col=0)

    # getting list of all Types and all Topics
    Types = set()
    Topics = set()
    for index, row in primer_df.iterrows():
        Types.add(  row["Type"]  )
        Topics.add( row["Topic"] )

    #print(Types)
    #print(Topics)

    SAVE_DIR = "generated_data"
    df = pd.DataFrame(columns=["Type", "Topic", "prompt", "response", "primer_type", "generated_reflection"] + list(hyperparameters.keys()))
    
    # doing the "iterated k-fold" cross validation
    for i, Type in tqdm(enumerate(sorted(Types), 1)):
        print("Type:", Type)
        
        # take advantage of the numbering of the Types
        primer_set = primer_df[ primer_df["Type"] <= Type ]
        data_set = primer_df[ primer_df["Type"] > Type ]

        primer_set = primer_set[["prompt", "response", "reflection"]].reset_index()
        data_set = data_set[["Type", "Topic", "prompt", "response"]].drop_duplicates().reset_index()

        # actually loading primers
        Primers = PrimerManager(primer_set, seed=hyperparameters["seed"])
        
        # running test
        output_df = k_fold(data_set, Primers, GPT2FR, hyperparameters, debug=args.debug, save_dir=SAVE_DIR)
    
        # saving df
        df.to_csv(f"iterative_k_fold/Type_iterated_{primer_file}_iteration_{i}")

    """
    for i, Topic in tqdm(enumerate(sorted(Topics), 1)):
        print("Topic:", Topic)
        
        # take advantage of the alphabetical nature of the Types
        primer_set = primer_df[ primer_df["Topic"] <= Topic ]
        data_set = primer_df[ primer_df["Topic"] > Topic ]

        primer_set = primer_set[["prompt", "response", "reflection"]].reset_index()
        data_set = data_set[["Type", "Topic", "prompt", "response"]].drop_duplicates().reset_index()

        # actually loading primers
        Primers = PrimerManager(primer_set, seed=hyperparameters["seed"])
        
        # running test
        output_df = k_fold(data_set, Primers, GPT2FR, hyperparameters, debug=args.debug, save_dir=SAVE_DIR)

        # saving df
        df.to_csv(f"iterative_k_fold/Topic_iterated_{primer_file}_iteration_{i}")
    """
