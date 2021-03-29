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
import traceback

# setting global seed for reproducability
INIT_SEED = 100
random.seed(INIT_SEED)
np.random.seed(INIT_SEED)
if not INIT_SEED is None:
    torch.manual_seed(INIT_SEED)

def static_primer_set(df, primer_df, GPT2FR, hyperparameters, debug=False, save_dir="."):

    N = hyperparameters["num_shots"]
    NUM_ITERATIONS = 1                  # number of iterations until we print results

    Log = ""
    output_df = pd.DataFrame(columns=["Type", "Topic", "prompt", "response", "primer_type", "primers", "gpt2_input", "generated_reflection", "beam"] + list(hyperparameters.keys()))
    try:
        for index, row in tqdm(df.iterrows()):
            
            # logging output
            if index % NUM_ITERATIONS == 0:
                Log += log_print()
                Log += log_print("------------------------------")
                Log += log_print(f"Iteration: {index}")
                Log += log_print("------------------------------")
                Log += log_print()
            
            # convert dataframe to list of strings
            examples = [GPT2FR.convert_example_to_formatted_string( ex_row["prompt"], ex_row["response"], ex_row["reflection"] ) \
                            for _, ex_row in primer_df.iterrows()]

            # convert row we want to generate a reflection of to a string
            query_string = GPT2FR.convert_example_to_formatted_string( row["prompt"], row["response"] )

            # generating reflection
            gpt2_input = "\n\n".join(examples + [query_string])

            # generating reflections
            print("Generating Reflection...")
            GPT2FR.update_hyperparameters(hyperparameters, hyperparameters["search_type"])
            generated_reflections = GPT2FR(gpt2_input)
            
            total_hp = hyperparameters.copy()
            total_hp.update(GPT2FR.hyperparameters)
            
            for i, generated_reflection in enumerate(generated_reflections, 1):
                # saving to dictionary
                hp_str = str(total_hp)
                output_df = output_df.append({
                    "Type": row["Type"], 
                    "Topic": row["Topic"], 
                    "prompt": row["prompt"], 
                    "response": row["response"], 
                    "primer_type": "no_primers", 
                    "primers": "\n\n".join(examples),
                    "gpt2_input": gpt2_input,
                    "generated_reflection": generated_reflection,
                    "beam": i,
                    **hyperparameters
                }, ignore_index=True)
                
            # logging output
            if index % NUM_ITERATIONS == 0:
                Log += log_print(primer_type)
                for i, example in enumerate(examples):
                    Log += log_print(f"{i+1} {example}")
                
                Log += log_print(query_string)
                Log += log_print()
                
                Log += log_print(str(generated_reflections))
                Log += log_print()

    except (KeyboardInterrupt, SystemExit):
        # This way the code will still save the data even if an interrupt occurs
        pass
    except Exception as e: 
        Log += "ERROR" + "\n"
        Log += str(e) + "\n"
        print()
        traceback.print_exc()
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
        "num_shots": 0,
        "seed": INIT_SEED,
        "search_type": "greedy"
    }

    """
    hyperparameters = {
        "num_shots": 3,
        "repetition_penalty": 1.0,
        "top_k": 50,
        "temperature": 0.5,
        "seed": INIT_SEED,
        "search_type": "greedy"
    }
    """

    print("\nLoading model...")
    GPT2FR = GPT2ForReflections(model_name=args.model)


    #primer_file = "simple_reflections_gpt2.csv"
    #primer_file = "complex_reflections_gpt2.csv"
    primer_file = "simple_reflections_human.csv"
    #primer_file = "complex_reflections_human.csv"
    data_set = pd.read_csv(f'static_data/{primer_file}', index_col=0)

    primer_df = pd.DataFrame(columns=["prompt", "response", "reflection"])

    SAVE_DIR = "generated_data"
    df = static_primer_set(data_set, primer_df, GPT2FR, hyperparameters, debug=args.debug, save_dir=SAVE_DIR)
