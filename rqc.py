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

    def convert_example_to_feature(self, s1, s2):
        # combine step for tokenization, WordPiece vector mapping, adding special tokens as well as truncating reviews longer than the max length
        return 

    # map to the expected input to TFBertForSequenceClassification, see here 
    @staticmethod
    def _to_dict(input_ids, attention_masks, token_type_ids):
        return {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
                "token_type_ids": token_type_ids,       # don't actually use these, but sometimes they are necessary
            }

    def preprocess_input(self, text_a, text_b):

        max_length = max([len(self.get_tokens(self.tokenizer, s1 + ' ' + s2)) for s1, s2 in zip(text_a, text_b)])
        #max_length = 512

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
                                    #return_tensors = 'pt',          # Return pytorch tensors
                                )
            
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            token_type_ids.append(encoded_dict['token_type_ids'])

        return tf.data.Dataset.from_tensor_slices((input_ids, attention_masks, token_type_ids)).map(self._to_dict)


    def predict(self, prompts, responses, reflections, batch_size=16):
        prompts = list(prompts)
        responses = list(responses)
        reflections = list(reflections)

        text_a = [self.get_prompt_response_string(prompt, response) for prompt, response in zip(prompts, responses)]
        text_b = reflections

        # to_predict is a TensorFlow object containing input_ids, attention_masks, and token_type_ids
        to_predict = self.preprocess_input(text_a, text_b)

        # calculating predictions using a batch size of batch_size
        predictions = self.model.predict(to_predict.batch(batch_size), verbose=True)

        # predictions is a TensorFlow object, so just extracting the data we care about
        logits = predictions.logits

        probs = tf.nn.softmax(logits)           # This is shape (N, 2) with the first column being P(label = 0) and second column being P(label = 1)
        labels = tf.math.argmax(probs, 1)       # the label is the argmax of each row

        confidence = probs[:, 1]                # "confidence" is a range between 0 and 1 while labels are boolean

        return confidence.numpy(), labels.numpy()

if __name__ == "__main__":
    
    import pandas as pd
    df  = pd.read_csv("../Variance Data/brute_force_SIMILAR_912.csv", index_col=0).fillna('')

    prompts = df["prompt"].values
    responses = df["response"].values
    reflections = df[df.columns[2]].values

    MODEL_PATH = "../distilbert_model_reflection_extended.08-0.85.hdf5"
    RQC = ReflectionQualityClassifier(MODEL_PATH)
    confidence, labels = RQC.predict(prompts, responses, reflections)
    print(confidence)
    print(labels)