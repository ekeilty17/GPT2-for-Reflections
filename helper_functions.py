def generate_reflection(prompt, response, Primers, GPT2FR, perm='default'):

    # Generating permutation
    num_shots = GPT2FR.hyperparameters["num_shots"]
    perm = list(range(num_shots))
    if perm != 'default':
        np.random.shuffle(perm)

    # Getting primers
    query_string = GPT2FR.convert_example_to_formatted_string(prompt, response)
    primer_examples = Primers.get_n_best_examples(query_string, num_shots)
    primer_examples = [GPT2FR.convert_example_to_formatted_string( ex_row["prompt"], ex_row["response"], ex_row["human"] ) \
                            for _, ex_row in primer_examples.iterrows()]

    # Getting gpt2 input
    primer_examples_permuted = [primer_examples[p] for p in perm]
    gpt2_input = "\n\n".join(primer_examples_permuted + [query_string])

    # Getting reflection
    return GPT2FR(gpt2_input)

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