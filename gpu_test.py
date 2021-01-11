
""" -------------------- Tensorflow -------------------- """

# useful links
# https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130
# https://www.tensorflow.org/install/gpu
# https://www.tensorflow.org/guide/gpu
# https://www.pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/

import tensorflow as tf
from tensorflow.python.client import device_lib

device_name = tf.test.gpu_device_name()

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
GPUs = get_available_gpus()

num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))

print("\n\n\n")
print("Tensorflow:")

print("\tVerions:", tf.__version__)
print("\tDevice name:", device_name) 
print( "\t", GPUs )
print("\tNum GPUs Available:", num_gpus)

""" -------------------- PyTorch -------------------- """

import torch

print("\n\n\n")
print("PyTorch:")

print("\tVerions:", torch.__version__)
print("\tIs GPU available?", torch.cuda.is_available())
print("\tNum GPUs Available:",  torch.cuda.device_count())
print(f"\tCurrent Device: cuda:{torch.cuda.current_device()}")
