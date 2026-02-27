# https://arxiv.org/pdf/2409.14013
import numpy as np

from data_proc import read_iot_data
from chronogan import chronogan

original_data = []
generated_data = []

DATA = "m5"
if DATA == "iot":
    data_df = read_iot_data()
    data = data_df.to_numpy()
if DATA == "m5":
    data = np.load("../../data/m5/m5_X_365.npz")["data"].squeeze()

# Split data
train_idx = int(0.8 * len(data))
train_data = data[:train_idx]
test_data = data[train_idx:]

original_data = train_data[..., np.newaxis]
original_data = original_data
print(original_data.shape)

## Newtork parameters
parameters = dict()

parameters['hidden_dim'] = 'same'
parameters['iterations'] = 10 # 6 * 10
parameters['batch_size'] = 128
parameters['num_layer'] = 4


chronogan_result = chronogan(original_data, parameters, 'same')
print(chronogan_result.shape)

np.savez_compressed(f'./output_{DATA}/synthetic_data.npz', data=chronogan_result)