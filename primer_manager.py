import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

# supress logging errors
import logging
tf.get_logger().setLevel(logging.ERROR)

class PrimerManager(object):

    def __init__(self, path='static_data/filtered_primers.csv', seed=None):
        self.primer_df = pd.read_csv(path, index_col=0)
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        
        # I pre-process the embeddings to save computation time
        prompt_response_strings = [self.get_prompt_response_string(row['prompt'], row['response']) for index, row in self.primer_df.iterrows()]
        self.primer_embeddings = self.embed(prompt_response_strings)

        self.seed = seed

    @staticmethod
    def get_prompt_response_string(prompt, response):
        return prompt + '\n' + response

    @staticmethod
    def consine_similarity(t1, t2, axis=-1):
        return tf.keras.losses.cosine_similarity(t1, t2, axis=axis)

    def get_n_random_examples(self, n):
        return self.primer_df.sample(n=n, random_state=self.seed)

    def get_n_best_examples(self, string, n):
        
        string_embedding = self.embed([string])[0]

        similarities = []
        for (index, _), primer_embedding in zip(self.primer_df.iterrows(), self.primer_embeddings):
            similarity = self.consine_similarity(string_embedding, primer_embedding)
            similarities.append( (index, float(similarity)) )
        
        similarities = list(sorted(similarities, key=lambda t: t[1]))
        return self.primer_df.iloc[ [index for index, _ in similarities[:n]] ]

if __name__ == "__main__":
    
    # test prompt-response
    prompt = "I appreciate you confirming my understanding OK, so smoking is pleasant and relaxing for you Are there other things that are good about smoking? If so, please tell me"
    response = "It  gives me a nice sensation."
    
    print(prompt)
    print(response)
    print()


    Primers = PrimerManager()
    query_string = Primers.get_prompt_response_string(prompt, response)
    

    print("Similar Primers")
    primer_examples = Primers.get_n_best_examples(query_string, 5)
    print(primer_examples)
    print()


    print("Random Primers")
    primer_examples = Primers.get_n_random_examples(5)
    print(primer_examples)