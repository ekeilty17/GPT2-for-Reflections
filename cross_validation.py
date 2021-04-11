from primer_manager import PrimerManager
from gpt2_for_reflections import GPT2ForReflections
from rqc import ReflectionQualityClassifier

from helper_functions import *
from generate_reflections_over_dataset import generate_reflections_over_dataset

import pandas as pd
from tqdm import tqdm

# setting global seed for reproducability
INIT_SEED = 100
set_global_seeds(INIT_SEED)

"""
I call this "Leave-One-Group-Out Cross Validation"
We have 2 groupings of the primer set: Topic and Types

We iterate over the groups, remove one, and reflect over those prompt/response pairs.
Repeat for each group
"""

def leave_one_group_out_cross_validation(df, Group_set, group_name, hyperparameters, primer_df, GPT2FR, save_dir='.'):
    for group in tqdm(Group_set):
        print(f"{group_name}:", group)
        
        primer_set = primer_df[ primer_df[group_name] != group ]
        data_set = primer_df[ primer_df[group_name] == group ]

        primer_set = primer_set[["prompt", "response", "reflection"]].reset_index()
        data_set = data_set.reset_index()

        # actually loading primers
        Primers = PrimerManager(primer_set, seed=hyperparameters["seed"])
        
        # running test
        output_df = generate_reflections_over_dataset(data_set, Primers, GPT2FR, hyperparameters, save_dir=SAVE_DIR)

        # saving output
        df = df.append(output_df, ignore_index=True)
    
    df.to_csv(f"{save_dir}/leave_{group_name}_out_cross_validation_{primer_file}")
    return df


if __name__ == "__main__":
    # parse commandline (helper_functions.py contains code)
    args = get_args()

    hyperparameters = {
        "num_shots": 5,
        "repetition_penalty": 1.0,
        "top_k": 10,
        "temperature": 0.1,
        "seed": INIT_SEED,
        "search_type": "sample"
    }

    print("\nLoading model...")
    GPT2FR = GPT2ForReflections(model_name=args.model)

    print("\nLoading Primers...")
    #primer_file = "simple_reflections_gpt2.csv"
    #primer_file = "complex_reflections_gpt2.csv"
    primer_file = "simple_reflections_human.csv"
    #primer_file = "complex_reflections_human.csv"
    primer_df = pd.read_csv(f'static_data/Final Thesis Primer Sets/{primer_file}', index_col=0)

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
    
    # doing the cross validation
    df = leave_one_group_out_cross_validation(df, Types,  "Type",  hyperparameters, primer_df, GPT2FR, save_dir=SAVE_DIR)
    #df = leave_one_group_out_cross_validation(df, Topics, "Topic", hyperparameters, primer_df, GPT2FR, save_dir=SAVE_DIR)
