## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings

from metrics.discriminative_metrics import discriminative_score_metrics
from data_proc import read_iot_data

warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan



def get_iot_data():
    ## Data loading
    data_df = read_iot_data()
    original_data = data_df.to_numpy()
    seq_length = original_data.shape[1]
    original_data = original_data[..., np.newaxis]

    return  original_data, seq_length

def train_data(original_data):
    ## Network parameters
    parameters = dict()

    parameters['module'] = 'gru' # or 'lstm', 'lstmLN'
    parameters['hidden_dim'] = 100
    parameters['num_layer'] = 3
    parameters['iterations'] = 10
    parameters['batch_size'] = 128

    # Run TimeGAN
    generated_data = timegan(original_data, parameters)
    print('Finish Synthetic Data Generation')

    return generated_data


def get_discriminative_score(original_data, generated_data):
    metric_iteration = 5

    discriminative_score = list()
    for _ in range(metric_iteration):
        temp_disc = discriminative_score_metrics(original_data, generated_data, iterations=50)
        discriminative_score.append(temp_disc)

    print('Discriminative score: ' + str(np.round(np.mean(discriminative_score), 4)))


if __name__ == "__main__":
    # original_data, seq_length = get_iot_data()
    # generated_data = train_data(original_data)
    # np.savez_compressed('./out/iot_timegan_generated_data.npz', data=generated_data)

    original_data = np.load("../../data/m5/m5_X_365.npz")["data"]
    generated_data = train_data(original_data)
    np.savez_compressed('./out/m5_timegan_generated_data.npz', data=generated_data)

    # generated_data = np.load('wm_timegan_generated_data.npz', allow_pickle=True)
    # generated_data = generated_data['data']

