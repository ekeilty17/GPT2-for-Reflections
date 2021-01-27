from re import search
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# hide the loading messages
import transformers
import logging
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

class GPT2ForReflections(object):

    def __init__(self, model_name='gpt2', search_type="greedy", hyperparameters=None):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # If anyone can find a source to find out what GPT2 does by default that would be amazing
        self.default_hyperparameters = {
            "max_len": 50,                  # default: N/A
            "repetition_penalty": 1.0,      # default: 1.0
            "num_beams": 5,                 # default: 1
            "num_return_sequences": 1,      # default: 1 (note: must be <= num_beams)
            "no_repeat_ngram_size": 0,      # default: 0
            "early_stopping": True,         # default: False
            "do_sample": True,              # default: False
            "top_k": 0,                     # default: 0?
            "top_p": 1.0,                   # default: 1.0?
            "temperature": 1.0              # default: 1?
        }
        
        # store user's hyper
        self.user_hyperparameters = {} if hyperparameters is None else {k: v for k, v in hyperparameters.items() if k in self.default_hyperparameters}
        
        # choose the ones we need based on the search type
        self.set_search_type(search_type)


    def update_hyperparameters(self, hyperparameters, search_type=None):
        self.user_hyperparameters.update(hyperparameters)
        self.set_search_type(self.search_type if search_type is None else search_type)
    

    def set_search_type(self, search_type):
        self.search_type = search_type.lower()
        if self.search_type == "greedy":
            self.do_greedy()
        elif self.search_type == "beam":
            self.do_beam()
        elif self.search_type == "sample" or self.search_type == "sampling":
            self.do_sampling()
        elif self.search_type == "custom":
            self.do_custom()
        else:
            raise ValueError(f"search type {self.search_type} is not supported. Must be 'greedy', 'beam', or 'sample'.")

    def do_greedy(self):
        # update default hp with user's hp
        self.hyperparameters = self.default_hyperparameters.copy()
        self.hyperparameters.update(self.user_hyperparameters)

        # only include the relevant hp's
        hp_list = ["max_len", "repetition_penalty"]
        self.hyperparameters = {hp: self.hyperparameters[hp] for hp in hp_list}
    
    def do_beam(self):
        # check that num_return_sequences <= num_beams
        if "num_beams" in self.user_hyperparameters and "num_return_sequences" in self.user_hyperparameters:
            assert( self.user_hyperparameters["num_beams"] >= self.user_hyperparameters["num_return_sequences"] )
        # If the user only changes num_return_sequences
        elif "num_return_sequences" in self.user_hyperparameters:
            # make sure num_beams >= to it
            if self.user_hyperparameters["num_return_sequences"] > self.default_hyperparameters["num_beams"]:
                # else update num_beams to match num_return_sequences
                self.default_hyperparameters["num_beams"] = self.user_hyperparameters["num_return_sequences"]
        
        # update default hp with user's hp
        self.hyperparameters = self.default_hyperparameters.copy()
        self.hyperparameters.update(self.user_hyperparameters)

        # only include the relevant hp's
        hp_list = ["max_len", "repetition_penalty", "num_beams", "num_return_sequences", "no_repeat_ngram_size", "early_stopping"]
        self.hyperparameters = {hp: self.hyperparameters[hp] for hp in hp_list}
        

    def do_sampling(self):
        if "temperature" in self.user_hyperparameters and "top_p" in self.user_hyperparameters:
            raise Exception("You can only specify temperature or top_p, but not both.")
        
        # update default hp with user's hp
        self.hyperparameters = self.default_hyperparameters.copy()
        self.hyperparameters.update(self.user_hyperparameters)

        # only include the relevant hp's
        hp_list = ["max_len", "repetition_penalty", "do_sample", "top_k"]
        if "top_p" in self.user_hyperparameters:
            hp_list += ["top_p"]
        else:
            hp_list += ["temperature"]
        self.hyperparameters = {hp: self.hyperparameters[hp] for hp in hp_list}

    def do_custom(self):
        self.hyperparameters = self.user_hyperparameters

    def __call__(self, text):
        return self.generate(text)

    def get_output(self, text):
        tokenized_text = self.tokenizer.encode(text, return_tensors="pt")
        tokenized_text = tokenized_text.to(self.device)
        summary_ids = self.model.generate(  tokenized_text,
                                            max_length=tokenized_text.shape[1] + self.hyperparameters["max_len"],
                                            bos_token_id=self.tokenizer.bos_token_id,
                                            pad_token_id=self.tokenizer.eos_token_id,
                                            **self.hyperparameters
                                    )
        output = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return output

    @staticmethod
    def convert_example_to_formatted_string(prompt, response, reflection='', delimiter='\n'):
        out  = f"Interviewer: {prompt}{delimiter}"
        out += f"Client: {response}{delimiter}"
        out += f"Reflection: {reflection}"
        return out
    
    @staticmethod
    def convert_to_formatted_input(examples, query_string, delimiter='\n\n'):
        return delimiter.join(examples + [query_string])

    @staticmethod
    def get_generated_output(gpt2_input, gpt2_output):
        return gpt2_output[len(gpt2_input):]
    
    @staticmethod
    def clean_reflection(generated_reflection):
        lines = generated_reflection.split('\n')
        return lines[0]

    def generate(self, text):
        output = self.get_output(text)
        generated_output = self.get_generated_output(text, output)
        generated_output = self.clean_reflection(generated_output)
        return generated_output

if __name__ == "__main__":

    # initializing model
    GPT2FR = GPT2ForReflections(model_name="gpt2")


    # get random prompt-response pair
    import pandas as pd
    df = pd.read_csv('static_data/filtered_primers.csv', index_col=0)

    # getting primers
    primer_row = df.iloc[0]
    primer_prompt, primer_response, primer_reflection = primer_row["prompt"], primer_row["response"], primer_row["reflection_human"]
    primer_string = GPT2FR.convert_example_to_formatted_string(primer_prompt, primer_response, primer_reflection)

    # getting query string
    query_row = df.iloc[1]
    query_prompt, query_response = query_row["prompt"], query_row["response"]
    query_string = GPT2FR.convert_example_to_formatted_string(query_prompt, query_response)
    
    # getting full GPT2 input
    gpt2_input = "\n\n".join([primer_string] + [query_string])
    print(gpt2_input)

    print("\n")


    # using the actual model
    reflection = GPT2FR(query_string)
    print("Generated Reflection:")
    print(reflection)