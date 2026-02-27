import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import os
import json
import pathlib
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import seaborn as sb

from torch.utils.tensorboard import SummaryWriter

# removed: import argparse

from ddpm import *
from data_make import *

import warnings

warnings.filterwarnings('ignore')






def visualize(ori_data, fake_data, dataset_name, seq_len, save_path, epoch, writer):
    ori_data = np.asarray(ori_data)

    fake_data = np.asarray(fake_data)

    ori_data = ori_data[:fake_data.shape[0]]

    sample_size = 250

    idx = np.random.permutation(len(ori_data))[:sample_size]

    randn_num = np.random.permutation(sample_size)[:1]

    real_sample = ori_data[idx]

    fake_sample = fake_data[idx]

    real_sample_2d = real_sample.reshape(-1, seq_len)

    fake_sample_2d = fake_sample.reshape(-1, seq_len)

    mode = 'visualization'

    ### PCA

    pca = PCA(n_components=2)
    pca.fit(real_sample_2d)
    pca_real = (pd.DataFrame(pca.transform(real_sample_2d))
                .assign(Data='Real'))
    pca_synthetic = (pd.DataFrame(pca.transform(fake_sample_2d))
                     .assign(Data='Synthetic'))
    pca_result = pca_real.append(pca_synthetic).rename(
        columns={0: '1st Component', 1: '2nd Component'})

    ### TSNE

    tsne_data = np.concatenate((real_sample_2d,
                                fake_sample_2d), axis=0)

    tsne = TSNE(n_components=2,
                verbose=0,
                perplexity=40)
    tsne_result = tsne.fit_transform(tsne_data)

    tsne_result = pd.DataFrame(tsne_result, columns=['X', 'Y']).assign(Data='Real')

    tsne_result.loc[len(real_sample_2d):, 'Data'] = 'Synthetic'

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))

    sb.scatterplot(x='1st Component', y='2nd Component', data=pca_result,
                   hue='Data', style='Data', ax=axs[0, 0])
    sb.despine()

    axs[0, 0].set_title('PCA Result')

    sb.scatterplot(x='X', y='Y',
                   data=tsne_result,
                   hue='Data',
                   style='Data',
                   ax=axs[0, 1])
    sb.despine()

    axs[0, 1].set_title('t-SNE Result')

    axs[1, 0].plot(real_sample[randn_num[0], :, :])

    axs[1, 0].set_title('Original Data')

    axs[1, 1].plot(fake_sample[randn_num[0], :, :])

    axs[1, 1].set_title('Synthetic Data')

    fig.suptitle('Assessing Diversity: Qualitative Comparison of Real and Synthetic Data Distributions',
                 fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=.88)

    plt.savefig(os.path.join(f'{save_path}', f'{time.time()}-tsne-result-{dataset_name}.png'))

    writer.add_figure(mode, fig, epoch)


def main():
    # -------------------------
    # CONFIG
    # -------------------------
    dataset_name = 'iot'         # choices: 'sine','stock','air','energy', 'iot'
    beta_schedule = 'cosine'    # choices: 'cosine','linear','quadratic','sigmoid'
    objective = 'pred_v'        # choices: 'pred_x0','pred_v','pred_noise'
    seq_len = 365
    batch_size = 16
    n_head = 2
    hidden_dim = 64
    num_of_layers = 3
    training_epoch = 10
    timesteps = 1

    # use globals from CONFIG section above
    seq = seq_len
    epochs = training_epoch
    timesteps_local = timesteps
    bs = batch_size
    latent_dim = hidden_dim
    num_layers = num_of_layers
    n_heads = n_head
    dataset = dataset_name
    beta_sched = beta_schedule
    obj = objective

    # load data
    train_data, test_data = LoadData(dataset, seq)

    train_data, test_data = np.asarray(train_data), np.asarray(test_data)

    features = train_data.shape[2]

    train_data, test_data = train_data.transpose(0, 2, 1), test_data.transpose(0, 2, 1)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=bs)

    test_loader = torch.utils.data.DataLoader(test_data, len(test_data))

    real_data = next(iter(test_loader))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mode = 'diffusion'

    architecture = 'custom-transformers'

    loss_mode = 'l1'

    file_name = f'{architecture}-{dataset}-{loss_mode}-{beta_sched}-{seq}-{obj}'

    folder_name = f'saved_files/{time.time():.4f}-{file_name}'

    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

    gan_fig_dir_path = f'{folder_name}/output/gan'

    pathlib.Path(gan_fig_dir_path).mkdir(parents=True, exist_ok=True)

    file_name_gan_fig = f'{file_name}-gan'

    # save params as a dict that you can edit above
    params_to_save = {
        'dataset_name': dataset,
        'beta_schedule': beta_sched,
        'objective': obj,
        'seq_len': seq,
        'batch_size': bs,
        'n_head': n_heads,
        'hidden_dim': latent_dim,
        'num_of_layers': num_layers,
        'training_epoch': epochs,
        'timesteps': timesteps_local
    }

    with open(f'{folder_name}/params.txt', 'w') as f:
        json.dump(params_to_save, f, indent=2)
        f.close()

    writer = SummaryWriter(log_dir=folder_name, comment=f'{file_name}', flush_secs=45)

    model = TransEncoder(

        features=features,
        latent_dim=latent_dim,
        num_heads=n_heads,
        num_layers=num_layers

    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length=seq,
        timesteps=timesteps_local,
        objective=obj,  # pred_x0, pred_v
        loss_type='l2',
        beta_schedule=beta_sched
    )

    diffusion = diffusion.to(device)

    lr = 1e-4

    betas = (0.9, 0.99)

    optim = torch.optim.Adam(diffusion.parameters(), lr=lr, betas=betas)

    for running_epoch in tqdm(range(epochs)):

        for i, data in enumerate(train_loader):

            data = data.to(device)

            batch_size_actual = data.shape[0]

            optim.zero_grad()

            loss = diffusion(data)

            loss.backward()

            optim.step()

            if i % len(train_loader) == 0:
                writer.add_scalar('Loss', loss.item(), running_epoch)

            if i % len(train_loader) == 0 and running_epoch % 100 == 0:
                print(f'Epoch: {running_epoch + 1}, Loss: {loss.item()}')

            if i % len(train_loader) == 0 and running_epoch % 500 == 0:
                with torch.no_grad():
                    samples = diffusion.sample(len(test_data))

                    samples = samples.cpu().numpy()

                    samples = samples.transpose(0, 2, 1)

                    np.save(f'{folder_name}/synth-{dataset}-{seq}-{running_epoch}.npy', samples)

                visualize(real_data.cpu().numpy().transpose(0, 2, 1), samples, dataset, seq, gan_fig_dir_path, running_epoch, writer)

    torch.save({
        'epoch': running_epoch + 1,
        'diffusion_state_dict': diffusion.state_dict(),
        'diffusion_optim_state_dict': optim.state_dict()

    }, os.path.join(f'{folder_name}', f'{file_name}-final.pth'))


if __name__ == "__main__":
    main()
