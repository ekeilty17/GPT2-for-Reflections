import pandas as pd
from tqdm import tqdm
from helper_functions import log_print
import traceback

""" This is the function that is doing all of the heavy lifting in the code """


def generate_reflections_over_dataset(df, Primers, GPT2FR, hyperparameters, permutations=1, debug=False, save_dir="."):

    N = hyperparameters["num_shots"]
    NUM_ITERATIONS = 1                  # number of iterations until we print results

    # TODO: implement permutations for any N
    assert N == 5, "Permutations will not work"

    PERMUTATIONS = [
                [0, 1, 2, 3, 4],
                [3, 1, 0, 4, 2],
                [4, 2, 1, 0, 3],
                [2, 0, 4, 3, 1],
                [1, 2, 0, 4, 3],
                [0, 3, 2, 1, 4],
                [4, 3, 2, 1, 0]
            ]

    Log = ""
    output_df = pd.DataFrame(columns=["Type", "Topic", "prompt", "response", "primer_type", "generated_reflection"] + list(hyperparameters.keys()))
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
                Log += log_print(f"Index: {index}")
                Log += log_print("------------------------------")
                Log += log_print()
            
            for examples, primer_type in zip([random_examples, similar_examples, different_examples], ["random", "similar", "different"]):

                for p in range(permutations):
                    # convert dataframe to list of strings
                    examples_formatted = [GPT2FR.convert_example_to_formatted_string( ex_row["prompt"], ex_row["response"], ex_row["reflection"] ) \
                                    for _, ex_row in examples.iterrows()]

                    # generating reflection
                    examples_permuted = [examples_formatted[i] for i in PERMUTATIONS[p]]
                    gpt2_input = "\n\n".join(examples_permuted + [query_string])

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
                            "primer_type": primer_type, 
                            "primers": "\n\n".join(examples_permuted),
                            "gpt2_input": gpt2_input,
                            "generated_reflection": generated_reflection,
                            "beam": i,
                            **hyperparameters
                        }, ignore_index=True)
                    
                    # logging output
                    if index % NUM_ITERATIONS == 0:
                        Log += log_print(primer_type)
                        for i, example in enumerate(examples_permuted):
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