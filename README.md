# Generating Reflections with GPT-2

This is the complete repository used to complete my undergraduate thesis.

There are 5 types of files/scripts you will find in this repository:
* Python Classes
    * `gpt2_for_reflections.py`
    * `rqc.py`
    * `primer_manager.py`
* Helper Functions
    * `gpu_test.py`
    * `helper_functions.py`
    * `generate_reflections_over_dataset.py`
    * `pipeline.py`
* Experimentation/Testing scripts
    * `hyperparameter_search.py`
    * `cross_validation.py`
    * `primer_set_expansion.py`
* Primer Data (anything found in `static_data/`)
* Reflection Data (anything found in `generated_data/`)

I will give a brief description of the purpose of each file

## Python Classes
* `gpt2_for_reflections.py`: This is a wrapper class over the standard PyTorch GPT2 model. One of the most important thing it handles is GPT-2's hyperparameters. Surprisingly, it's hard to find information about how GPT-2 uses each hyperparameters. Reading over this class should give you better insight as to what each hyperparameter does. This class also handles properly formatting of the few-short learning input, which has some subtle nuances to it. Finally, it handles the output of GPT-2, which can be different depending on the decoding type.

* `rqc.py`: RQC stands for "Reflection Quality Classifier". This is a fine-tuned BERT model that was trained to classify whether or not a reflection to a given prompt/response pair is good. This class is a wrapper class over the model to make getting the reflection labels much easier for the user.

* `primer_manager.py`: As the name suggests, this class manages the primers. The class contains methods which implement different primer-selection methods. Currently the similar, random, and different methods have been implemented, utilizing TensorFlow's universal sentence encoder.


## Helper Functions
* `gpu_test.py`: Running this script will simply tell you if you have the correct drivers installed on your machine or virtual machine in order to use a GPU with PyTorch and TensorFlow. 

* `helper_functions.py`: This is just a random mix of useful functions. Looking at the code will tell you all you need about what these functions do.

* `generate_reflections_over_dataset.py`: This is the work-horse of the entire repository. Every Experimentation/Testing script utilizes this function. It is essentially iterating over the entire dataset and producing a reflection with GPT-2 utilizing various primer selection methods. There are a lot of nuances in this script and is the result of almost a year of iteration. So if you have any questions please contact me, and I am more than happy to walk through the code.

* `pipeline.py`: This is putting everything together. This is the skeleton of the script you would use to deploy the GPT-2 model.


## Experimentation/Testing Scripts
* `hyperparameter_search.py`: This script preforms a grid-search over various hyperparameters specified by the user.

* `cross_validation.py`: This script preforms the "Leave-One-Group-Out Cross Validation" experiment. For details as to what that is, please read my thesis.

* `primer_set_expansion.py`: This script preforms the "Primer-Set Expansion" experiment. For details as to what that is, please read my thesis.
