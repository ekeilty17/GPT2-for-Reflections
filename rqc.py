import numpy as np

import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

# hide the loading messages
import transformers
import logging
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

class ReflectionQualityClassifier(object):

    def __init__(self, path, model_name="distilbert-base-uncased"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = TFDistilBertForSequenceClassification.from_pretrained(model_name)
        self.model.load_weights(path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)


    @staticmethod
    def get_tokens(tokenizer, string):
        return tokenizer.tokenize(string)


    # I believe this is exactly the same as calling tokenizer.encode()
    @staticmethod
    def get_token_ids(tokenizer, string):
        return tokenizer.convert_tokens_to_ids( tokenizer.tokenize(string) )


    # for this task, we only have text_a no text_b
    def preprocess_input(self, text_a):

        max_length = max([len(self.get_tokens(self.tokenizer, sentence)) for sentence in text_a])

        input_ids = []
        attention_masks = []
        token_type_ids = []
        for sentence in text_a:
            encoded_dict = self.tokenizer.encode_plus(
                                    sentence,                       # Sentence to encode
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
        
        return input_ids, attention_masks, token_type_ids


    def predict(self, text_a):

        input_ids, attention_masks, token_type_ids = self.preprocess_input(text_a)

        logits = self.model.predict([
            input_ids.numpy(), 
            attention_masks.numpy()
        ])[0]

        predictions = np.argmax(logits, axis=1)
        return predictions