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


# setting seed for reproducability
SEED = 100
random.seed(SEED)
np.random.seed(SEED)
if not SEED is None:
    torch.manual_seed(SEED)


def grid_search(df, Primers, GPT2FR, hyperparameters_dict, debug=False, save_dir="."):

    # the logging and saving doesn't really work as well if there are multiple search types
    if "search_type" in hyperparameters_dict:
        if type(hyperparameters_dict["search_type"]) == list:
            if len(hyperparameters_dict["search_type"]) > 1:
                raise ValueError("Please only select 1 value for hyperparameter 'search_type'")

    # pre-processing the hyperparameter dictionary to make things easier to iterate over
    hyperparameters = {key: val if type(val) == list else [val] for key, val in hyperparameters_dict.items()}

    #       do that by taking the current dictionary of lists and convert that to a list of dictionaries
    #       to do that, we need the cartensian product of every list in the original dictionary
    hp_names = hyperparameters.keys()
    hp_combinations = itertools.product(*hyperparameters.values())
    hyperparameter_list = [ dict(zip(hp_names, comb)) for comb in hp_combinations ]

    # holds all reflections, will be added to df later
    #       I couldn't think of a better unique identifier other than just using the string version of the hyperparmater dictionary
    #       I could have used a tuple of it, but then we lose the names
    generated_reflection_by_hyperparameter = {}
    for hp in hyperparameter_list:
        GPT2FR.update_hyperparameters(hp, hp["search_type"])
        total_hp = hp.copy()
        total_hp.update(GPT2FR.hyperparameters)
        hp_str = str(total_hp)
        generated_reflection_by_hyperparameter[hp_str] = []

    print("Hyperparameter set...")
    for hp in generated_reflection_by_hyperparameter:
        print(hp)

    # since we've added default values, we need to update this variable to use later
    hp_names = list(ast.literal_eval(list(generated_reflection_by_hyperparameter.keys())[0]).keys())

    # Log string
    Log = ""

    SEED = hyperparameters["seed"]
    
    # calculating a static set of random examples
    N = hyperparameters_dict["num_shots"] if type(hyperparameters_dict["num_shots"]) == int else max(hyperparameters_dict["num_shots"])
    random_examples = Primers.get_n_random_examples(N)

    print(random_examples)

    # a try-except statement so you can do control C to stop early and it will save the progress
    try:
        for index, row in tqdm(df.iterrows()):

            #random_examples = Primers.get_n_random_examples(N)
            best_examples = Primers.get_n_best_examples(Primers.get_prompt_response_string(row["prompt"], row["response"]), N)

            # convert row we want to generate a reflection of to a string
            query_string = GPT2FR.convert_example_to_formatted_string( row["prompt"], row["response"] )

            for hp in hyperparameter_list:
                #print(hp)
                
                examples = random_examples[:hp["num_shots"]] if hp["random"] else best_examples[:hp["num_shots"]]

                # convert dataframe to list of strings
                examples = [GPT2FR.convert_example_to_formatted_string( ex_row["prompt"], ex_row["response"], ex_row["reflection_human"] ) \
                                for _, ex_row in examples.iterrows()]

                # generating reflection
                gpt2_input = "\n\n".join(examples + [query_string])

                # generating reflections
                print("Generating Reflection...")
                GPT2FR.update_hyperparameters(hp, hp["search_type"])
                generated_reflections = GPT2FR(gpt2_input)

                total_hp = hp.copy()
                total_hp.update(GPT2FR.hyperparameters)

                # saving to dictionary
                hp_str = str(total_hp)
                generated_reflection_by_hyperparameter[hp_str].append(generated_reflections)
                
            # logging output
            NUM_ITERATIONS = 1                  # number of iterations until we print results
            if index % NUM_ITERATIONS == 0:
                Log += log_print()
                Log += log_print("------------------------------")
                Log += log_print(f"Iteration: {index}")
                Log += log_print("------------------------------")
                Log += log_print()
                
                for i, example in enumerate(examples):
                    Log += log_print(f"{i+1} {example}")
                Log += log_print(query_string)
                Log += log_print()
                
                Log += log_print(f"hyperparmater names: {list(hp_names)}")
                for hp_str, reflections in generated_reflection_by_hyperparameter.items():
                    hp = ast.literal_eval(hp_str)
                    Log += log_print(f"{str(list(hp.values())):50s}: {reflections[-1]}")
                Log += log_print()


    except (KeyboardInterrupt, SystemExit):
        # This way the code will still save the data even if an interrupt occurs
        pass
    except Exception as e: 
        Log += "ERROR" + "\n"
        Log += str(e) + "\n"
        print()
        raise Exception(e)
    
    # saving to dataframe
    for hp_str, reflections in generated_reflection_by_hyperparameter.items():
        df = add_column_to_dataframe(df, reflections, hp_str)

    # saving log file
    print("Saving log file...")
    with open(f"{save_dir}/Log.txt", "w+") as g:
        g.write(Log)

    return df

if __name__ == "__main__":
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Testing Simple Reflection Generation')
    parser.add_argument('--model', type=str, default='gpt2', help="Model name")
    parser.add_argument('--debug', action="store_true", default=False)
    args = parser.parse_args()

    
    for seed in [2020, 912, 1554]:
        hyperparameters = {
            "num_shots": 5,
            "repetition_penalty": 1.0,
            "top_k": [0, 10, 100],
            "temperature": [0.01, 0.1, 0.5, 1.0],
            "seed": seed,
            #"seed": np.random.randint(2020),
            "random": False,
            "search_type": "sample"
        }

        df = pd.read_csv('static_data/filtered_prompt_response_pairs.csv', index_col=0)

        print("\nLoading Primers...")
        Primers = PrimerManager()

        print("\nLoading model...")
        GPT2FR = GPT2ForReflections(model_name=args.model)

        
        print("\nRunning hyperparameter search...")
        SAVE_DIR = "generated_data"
        df = grid_search(df, Primers, GPT2FR, hyperparameters, debug=args.debug, save_dir=SAVE_DIR)

        print("\nSaving to csv...")
        primer_type = "RANDOM" if hyperparameters["random"] else "SIMILAR"
        df.to_csv(f"{SAVE_DIR}/brute_force_{primer_type}_{hyperparameters['seed']}.csv")
        #df.to_csv(f"{SAVE_DIR}/brute_force_RANDOM{k}.csv")
