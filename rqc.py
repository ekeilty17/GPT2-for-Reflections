import numpy as np

import torch

import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

# hide the loading messages
import transformers
import logging
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

class ReflectionQualityClassifier(object):

    """ How the model works

    Recall that when you use BERT you have text_a and text_b. 
    This was trained so that 
        text_a = prompt/response string (separated by a space)
        text_b = reflection string

    It outputs the logits, which are the values before you softmax
    The logits have shape (N, 2) where the 2 outputs represent
        ( P(good reflection), P(bad reflection) )
    """

    def __init__(self, path, model_name="distilbert-base-uncased"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = TFDistilBertForSequenceClassification.from_pretrained(model_name)
        self.model.load_weights(path)
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.model = self.model.to(self.device)


    @staticmethod
    def get_tokens(tokenizer, string):
        return tokenizer.tokenize(string)


    # I believe this is exactly the same as calling tokenizer.encode()
    @staticmethod
    def get_token_ids(tokenizer, string):
        return tokenizer.convert_tokens_to_ids( tokenizer.tokenize(string) )


    @staticmethod
    def get_prompt_response_string(prompt, response):
        return prompt + ' ' + response

    def preprocess_input(self, text_a, text_b):

        max_length = max([len(self.get_tokens(self.tokenizer, s1 + ' ' + s2)) for s1, s2 in zip(text_a, text_b)])

        input_ids = []
        attention_masks = []
        token_type_ids = []
        for s1, s2 in zip(text_a, text_b):
            encoded_dict = self.tokenizer.encode_plus(
                                    s1, s2,                         # Sentence to encode
                                    add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
                                    #padding='longest',             # Pad to longest input in batch
                                    padding = 'max_length',         # Pad to max sentence length
                                    max_length = max_length,   
                                    truncation = True,              
                                    return_attention_mask = True,   # Construct attn. masks
                                    return_token_type_ids = True,   # sometimes these are necessary
                                    return_tensors = 'pt',          # Return pytorch tensors
                                )
            
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            token_type_ids.append(encoded_dict['token_type_ids'])
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)

        return input_ids, attention_masks, token_type_ids


    def predict(self, prompts, responses, reflections):
        prompts = list(prompts)
        responses = list(responses)
        reflections = list(reflections)

        text_a = [self.get_prompt_response_string(prompt, response) for prompt, response in zip(prompts, responses)]
        text_b = reflections

        input_ids, attention_masks, token_type_ids = self.preprocess_input(text_a, text_b)

        logits = self.model.predict([
            input_ids.numpy(), 
            attention_masks.numpy()
        ])[0]

        print(logits.shape)

        labels = np.argmin(logits, axis=1)

        return labels

if __name__ == "__main__":
    
    import pandas as pd
    df  = pd.read_csv("../Variance Data/brute_force_SIMILAR_912.csv", index_col=0).fillna('')

    prompts = df["prompt"].values
    responses = df["response"].values
    reflections = df[df.columns[2]].values

    MODEL_PATH = "../distilbert_model_reflection_extended.08-0.85.hdf5"
    RQC = ReflectionQualityClassifier(MODEL_PATH)
    labels = RQC.predict(prompts, responses, reflections)
    print(labels)