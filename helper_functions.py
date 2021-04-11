
def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    if not seed is None:
        torch.manual_seed(seed)


def add_column_to_dataframe(df, data, column_name):
    if len(data) > len(df):
        data = data[:len(df)]
    elif len(data) < len(df):
        data += [''] * (len(df) - len(data))
    
    df.insert(len(df.columns), column_name, data)
    return df

def log_print(string=''):
    print(string)
    return string + '\n'

# same function found in pipline.py
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