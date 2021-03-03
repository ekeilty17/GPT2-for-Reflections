from primer_manager import PrimerManager
from gpt2_for_reflections import GPT2ForReflections
from rqc import ReflectionQualityClassifier

from helper_functions import *

import numpy as np
import pandas as pd
import torch
import random

from tqdm import tqdm
import itertools
import ast

import argparse

# setting global seed for reproducability
INIT_SEED = 100
random.seed(INIT_SEED)
np.random.seed(INIT_SEED)
if not INIT_SEED is None:
    torch.manual_seed(INIT_SEED)

def k_fold(df, Primers, GPT2FR, hyperparameters, debug=False, save_dir="."):

    N = hyperparameters["num_shots"]
    NUM_ITERATIONS = 1                  # number of iterations until we print results

    Log = ""
    output_df = pd.DataFrame(columns=["Type", "Topic", "prompt", "response", "primer_type", "generated_reflection"])
    try:
        for index, row in tqdm(df.iterrows()):

            # getting primers
            random_examples = Primers.get_n_random_examples(N)
            similar_examples = Primers.get_n_similar_examples(Primers.get_prompt_response_string(row["prompt"], row["response"]), N)
            different_examples = Primers.get_n_different_examples(Primers.get_prompt_response_string(row["prompt"], row["response"]), N)

            # convert row we want to generate a reflection of to a string
            query_string = GPT2FR.convert_example_to_formatted_string( row["prompt"], row["response"] )

            # logging output
            if index % NUM_ITERATIONS == 0:
                Log += log_print()
                Log += log_print("------------------------------")
                Log += log_print(f"Iteration: {index}")
                Log += log_print("------------------------------")
                Log += log_print()
            
            for examples, primer_type in zip([random_examples, similar_examples, different_examples], ["random", "similar", "different"]):

                # convert dataframe to list of strings
                examples = [GPT2FR.convert_example_to_formatted_string( ex_row["prompt"], ex_row["response"], ex_row["reflection_human"] ) \
                                for _, ex_row in examples.iterrows()]

                # generating reflection
                gpt2_input = "\n\n".join(examples + [query_string])

                # generating reflections
                print("Generating Reflection...")
                GPT2FR.update_hyperparameters(hyperparameters, hyperparameters["search_type"])
                generated_reflections = GPT2FR(gpt2_input)

                total_hp = hyperparameters.copy()
                total_hp.update(GPT2FR.hyperparameters)

                # saving to dictionary
                hp_str = str(total_hp)
                output_df = output_df.append({
                    "Type": row["Type"], 
                    "Topic": row["Topic"], 
                    "prompt": row["prompt"], 
                    "response": row["response"], 
                    "primer_type": primer_type, 
                    "generated_reflection": generated_reflection
                }, ignore_index=True)
                
                # logging output
                if index % NUM_ITERATIONS == 0:
                    Log += log_print(primer_type)
                    for i, example in enumerate(examples):
                        Log += log_print(f"{i+1} {example}")
                    
                    Log += log_print(query_string)
                    Log += log_print()
                    
                    Log += log_print(generated_reflection)
                    Log += log_print()

    except (KeyboardInterrupt, SystemExit):
        # This way the code will still save the data even if an interrupt occurs
        pass
    except Exception as e: 
        Log += "ERROR" + "\n"
        Log += str(e) + "\n"
        print()
        raise Exception(e)

    # saving log file
    print("Saving log file...")
    with open(f"{save_dir}/Log.txt", "w+") as g:
        g.write(Log)

    return output_df


if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Testing Simple Reflection Generation')
    parser.add_argument('--model', type=str, default='gpt2', help="Model name")
    parser.add_argument('--debug', action="store_true", default=False)
    args = parser.parse_args()

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
    primer_df = pd.read_csv('static_data/filtered_golden_primers.csv', index_col=0)

    # getting list of all Types and all Topics
    Types = set()
    Topics = set()
    for index, row in primer_df.iterrows():
        Types.add(  row["Type"]  )
        Topics.add( row["Topic"] )

    #print(Types)
    #print(Topics)

    SAVE_DIR = "generated_data"
    df = pd.DataFrame(columns=["Type", "Topic", "prompt", "response", "primer_type", "generated_reflection"])
    
    # doing the "k-fold" cross validation
    for Type in Types:
        if Type != 11:
            continue 
        
        primer_set = primer_df[ primer_df["Type"] != Type ]
        data_set = primer_df[ primer_df["Type"] == Type ]

        primer_set = primer_set[["prompt", "response", "reflection_human"]].reset_index()
        data_set = data_set[["Type", "Topic", "prompt", "response"]].drop_duplicates().reset_index()

        # actually loading primers
        Primers = PrimerManager(primer_set, seed=hyperparameters["seed"])
        
        # running test
        output_df = k_fold(data_set, Primers, GPT2FR, hyperparameters, debug=args.debug, save_dir=SAVE_DIR)

        # saving output
        df = df.append(output_df, ignore_index=True)
    
    df.to_csv(f"{SAVE_DIR}/Type_k_fold.csv")