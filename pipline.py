from primer_manager import PrimerManager
from gpt2_for_reflections import GPT2ForReflections
from rqc import ReflectionQualityClassifier

import pandas as pd
import numpy as np

""" Putting Everything Together """
def generate_reflection(prompt, response, Primers, GPT2FR, perm='default', num_shots=5):

    # Generating permutation
    perm = list(range(num_shots))
    if perm != 'default':
        np.random.shuffle(perm)

    # Getting primers
    primer_examples = Primers.get_n_similar_examples(Primers.get_prompt_response_string(prompt, response), num_shots)
    
    # converting primers to GPT2 input format
    query_string = GPT2FR.convert_example_to_formatted_string(prompt, response)
    primer_examples = [GPT2FR.convert_example_to_formatted_string( ex_row["prompt"], ex_row["response"], ex_row["reflection"] ) \
                            for _, ex_row in primer_examples.iterrows()]

    # Getting gpt2 input
    primer_examples_permuted = [primer_examples[p] for p in perm]
    gpt2_input = "\n\n".join(primer_examples_permuted + [query_string])

    # Getting reflection
    return GPT2FR(gpt2_input)

def is_good_reflection(prompt, response, reflection, RQC):
    confidence, labels = RQC.predict([prompt], [response], [reflection])

    # TODO: could also have a condition like confidence[0] > threshold
    return labels[0] == 1

# What the user calls
def get_good_reflection(prompt, response, Primers, GPT2FR, RQC, num_shots=5, max_iter=10):
    
    cnt = 0
    while True and cnt < max_iter:
        candidate_reflection = generate_reflection(prompt, response, Primers, GPT2FR, perm="random", num_shots=num_shots)
        if is_good_reflection(prompt, response, candidate_reflection, RQC):
            return candidate_reflection
        cnt += 1


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

    # loading primers
    primer_df = pd.read_csv("static_data/Final Thesis Primer Sets/complex_reflections_human.csv", index_col=0)
    Primers = PrimerManager(primer_df)
    
    # loading GPT2 model
    GPT2FR = GPT2ForReflections(model_name="gpt2", hyperparameters=hyperparameters)
    
    # loading RQC
    MODEL_PATH = "../bert_base_uncased_reflection_v2.1_balanced.08-0.80.hdf5"
    RQC = ReflectionQualityClassifier(MODEL_PATH, model_name="bert-base-uncased")

    prompt = "I appreciate you confirming my understanding OK, so smoking is pleasant and relaxing for you Are there other things that are good about smoking? If so, please tell me"
    response = "It  gives me a nice sensation."
    reflection = get_good_reflection(prompt, response, Primers, GPT2FR, RQC, num_shots=hyperparameters["num_shots"])
    print(reflection)