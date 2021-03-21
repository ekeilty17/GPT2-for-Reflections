from primer_manager import PrimerManager
from gpt2_for_reflections import GPT2ForReflections
from rqc import ReflectionQualityClassifier

import pandas as pd
import numpy as np

""" Putting Everything Together """

def convert_example_to_formatted_string(inp, reflection='', delimiter='\n'):
    prompt, response = inp

    out  = f"Interviewer: {prompt}{delimiter}"
    out += f"Client: {response}{delimiter}"
    out += f"Reflection: {reflection}"
    return out

def generate_reflection(prompt, response, Primers, GPT2FR, perm='default'):

    # Generating permutation
    num_shots = GPT2FR.hyperparameters["num_shots"]
    perm = list(range(num_shots))
    if perm != 'default':
        np.random.shuffle(perm)

    # Getting primers
    query_string = convert_example_to_formatted_string( (prompt, response) )
    primer_examples = Primers.get_n_best_examples(query_string, num_shots)
    primer_examples = [convert_example_to_formatted_string( (ex_row["prompt"], ex_row["response"]), ex_row["reflection"] ) \
                            for _, ex_row in primer_examples.iterrows()]

    # Getting gpt2 input
    primer_examples_permuted = [primer_examples[p] for p in perm]
    gpt2_input = "\n\n".join(primer_examples_permuted + [query_string])

    # Getting reflection
    return GPT2FR(gpt2_input)

# TODO
def is_good_reflection(reflection, RQC):
    return True

# What the user calls
def get_good_reflection(prompt, response, Primers, GPT2FR, RQC):
    
    while True:
        candidate_reflection = generate_reflection(prompt, response, Primers, GPT2FR, perm="random")
        if is_good_reflection(candidate_reflection, RQC):
            return candidate_reflection


# Example of how things would be called
if __name__ == "__main__":

    hyperparameters = {
        "num_shots": 5,
        "temperature": 0.175,
        "repetition_penalty": 1.0,
        "top_k": 100,
        "top_p": 0.8,
        "seed": None
    }

    Primers = PrimerManager()
    GPT2FR = GPT2ForReflections(model_name="gpt2", hyperparameters=hyperparameters)
    RQC = None#ReflectionQualityClassifier()

    prompt = "I appreciate you confirming my understanding OK, so smoking is pleasant and relaxing for you Are there other things that are good about smoking? If so, please tell me"
    response = "It  gives me a nice sensation."
    reflection = get_good_reflection(prompt, response, Primers, GPT2FR, RQC)
    print(reflection)