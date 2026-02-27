# https://github.com/AlexanderVNikitin/tsgm/
# https://github.com/AlexanderVNikitin/tsgm/blob/main/tutorials/GANs/TimeGAN.ipynb
# https://github.com/AlexanderVNikitin/tsgm/blob/main/tutorials/evaluation.ipynb
import json

import numpy as np
import tsgm
from matplotlib import pyplot as plt

from constants import RES_FOLDER
from data_proc import read_iot_data
from tsgm_plot import plot_real_vs_generated

# data_df = read_iot_data()
# data = data_df.to_numpy()
data = np.load("../../data/m5/m5_X_365.npz")["data"].squeeze()


# Split data
train_idx = int(0.8 * len(data))
train_data = data[:train_idx]
test_data = data[train_idx:]

train_data = train_data[..., np.newaxis]

scaler = tsgm.utils.TSFeatureWiseScaler()
scaled_data = scaler.fit_transform(train_data)

model = tsgm.models.timeGAN.TimeGAN(
    seq_len=train_data.shape[1],
    module="gru",
    hidden_dim=24,
    n_features=train_data.shape[2],
    n_layers=3,
    batch_size=256,
    gamma=1.0,
)
# .compile() sets all optimizers to Adam by default
model.compile()

model.fit(
    data=scaled_data,
    epochs=10,
)

n_samples_to_generate = 10
generated_data = model.generate(n_samples=len(scaled_data))
generated_data = np.array(generated_data)
print(generated_data.shape)
# np.save("tsgm_timegan_gen.npy", generated_data)
np.savez_compressed(RES_FOLDER + 'tsgm_timegan_gen.npz', data=generated_data)

tsgm.utils.visualize_original_and_reconst_ts(scaled_data, generated_data, num=n_samples_to_generate)
plt.savefig(RES_FOLDER + "tsgm_timegan_result.png")
plt.close()

plot_real_vs_generated(scaled_data, generated_data, n_display=10, savepath=RES_FOLDER + "tsgm_timegan_result_custom.png")



model_name = "tsgm_timegan"
losses = model.training_losses_history

with open(RES_FOLDER + f"{model_name}_losses.json", "w") as f:
    json.dump(losses, f)

for k,v in losses.items():
    print(f"{k}: {v}")