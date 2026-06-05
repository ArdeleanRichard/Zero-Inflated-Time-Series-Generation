import numpy as np

import tensorflow as tf

from repos.ChronoGAN.data_loading import real_data_loading, sine_data_generation

tf.compat.v1.disable_eager_execution()
import logging
tf.get_logger().setLevel(logging.ERROR)

# 1. ChronoGAN model
from chronogan import chronogan

original_data = []
generated_data = []

# ## Data loading
# data_name = 'sine'
# seq_len = 64
#
# if data_name == 'sine':
#     # Set number of samples and its dimensions
#     no, dim = 10000, 4
#     original_data.append(sine_data_generation(no, seq_len, dim))
#
# print(data_name + ' dataset is ready.')

## Data loading
# data_name = 'ECG'
# seq_len = 140

data_name = 'stock'
seq_len = 24

if data_name in ['stock', 'electricity', 'ECG']:
  original_data.append(real_data_loading(data_name, seq_len))

print(data_name + ' dataset is ready.')


## Newtork parameters
parameters = dict()

parameters['hidden_dim'] = 'same'
parameters['iterations'] = 6 * 1000
parameters['batch_size'] = 128
parameters['num_layer'] = 4

chronogan_result = chronogan(original_data[0], parameters, 'same')