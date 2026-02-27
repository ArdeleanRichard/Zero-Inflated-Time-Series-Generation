# https://github.com/AlexanderVNikitin/tsgm/
# https://github.com/AlexanderVNikitin/tsgm/blob/main/tutorials/VAEs/VAE.ipynb
# https://github.com/AlexanderVNikitin/tsgm/blob/main/tutorials/evaluation.ipynb
import functools
import json
import os

import numpy as np
from matplotlib import pyplot as plt

from constants import RES_FOLDER
from tsgm_plot import plot_real_vs_generated

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tsgm

from data_proc import read_iot_data


# data_df = read_iot_data()
# data = data_df.to_numpy()
data = np.load("../../data/m5/m5_X_365.npz")["data"].squeeze()


# Split data
train_idx = int(0.8 * len(data))
train_data = data[:train_idx]
test_data = data[train_idx:]

train_data = train_data[..., np.newaxis]

architecture = tsgm.models.zoo["vae_conv5"](train_data.shape[1], train_data.shape[2], 10)
encoder, decoder = architecture.encoder, architecture.decoder

scaler = tsgm.utils.TSFeatureWiseScaler()
scaled_data = scaler.fit_transform(train_data)

model = tsgm.models.cvae.BetaVAE(encoder, decoder)
model.compile(optimizer=keras.optimizers.Adam())

model.summary()

model.fit(scaled_data, epochs=10, batch_size=64)




n_samples_to_generate = 10
x_decoded = model.predict(scaled_data)
generated_data = model.generate(len(scaled_data))
generated_data = np.array(generated_data)
print(generated_data.shape)
# np.save("tsgm_vae_gen.npy", generated_data)
np.savez_compressed(RES_FOLDER + 'tsgm_vae_gen.npz', data=generated_data)

tsgm.utils.visualize_original_and_reconst_ts(scaled_data, generated_data, num=n_samples_to_generate)
plt.savefig(RES_FOLDER + "tsgm_vae_result.png")
plt.close()

plot_real_vs_generated(scaled_data, generated_data, n_display=n_samples_to_generate, savepath=RES_FOLDER + "tsgm_vae_result_custom.png")

# calculate the distance between a vector of summary statistics of synthetic data and real data
statistics = [
    functools.partial(tsgm.metrics.statistics.axis_max_s, axis=None),
    functools.partial(tsgm.metrics.statistics.axis_min_s, axis=None),
    functools.partial(tsgm.metrics.statistics.axis_max_s, axis=1),
    functools.partial(tsgm.metrics.statistics.axis_min_s, axis=1)
]

discrepancy_func = lambda x, y: np.linalg.norm(x - y)

dist_metric = tsgm.metrics.DistanceMetric(
    statistics=statistics, discrepancy=discrepancy_func
)
print(dist_metric(scaled_data, generated_data))

# Maximum Mean Discrepancy # OOM when allocating tensor with shape[39365,39365,365,1] and
# mmd_metric = tsgm.metrics.MMDMetric()
# print(mmd_metric(scaled_data, x_decoded))

model_name = "tsgm_vae"
total_loss = model.total_loss_tracker.result().numpy()
reconstruction_loss = model.reconstruction_loss_tracker.result().numpy()
kl_loss = model.kl_loss_tracker.result().numpy()

losses = {
    "total_loss": float(total_loss),
    "reconstruction_loss": float(reconstruction_loss),
    "kl_loss": float(kl_loss)
}

# Save all losses in a compressed file
np.savez_compressed(RES_FOLDER + f"{model_name}_losses.npz",
                    total_loss=total_loss,
                    reconstruction_loss=reconstruction_loss,
                    kl_loss=kl_loss)

with open(RES_FOLDER + f"{model_name}_losses.json", "w") as f:
    json.dump(losses, f)

print("Total loss:", total_loss)
print("Reconstruction loss:", reconstruction_loss)
print("KL loss:", kl_loss)
