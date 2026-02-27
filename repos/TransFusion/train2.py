import matplotlib.pyplot as plt
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
    pca_result = pd.concat([pca_real, pca_synthetic], ignore_index=True).rename(
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


def main_train(dataset_name):
    # -------------------------
    # CONFIG
    # -------------------------
    # dataset_name = 'iot'         # choices: 'sine','stock','air','energy', 'iot'
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

    # Track losses
    loss_history = []

    for running_epoch in tqdm(range(epochs)):

        epoch_losses = []

        for i, data in enumerate(train_loader):

            data = data.to(device)

            batch_size_actual = data.shape[0]

            optim.zero_grad()

            loss = diffusion(data)

            loss.backward()

            optim.step()

            epoch_losses.append(loss.item())

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

        # Store average loss for this epoch
        loss_history.append({'epoch': running_epoch + 1, 'loss': np.mean(epoch_losses)})

    # Save model
    torch.save({
        'epoch': running_epoch + 1,
        'diffusion_state_dict': diffusion.state_dict(),
        'diffusion_optim_state_dict': optim.state_dict()

    }, os.path.join(f'{folder_name}', f'{file_name}-final.pth'))

    # Save loss history as JSON
    with open(f'{folder_name}/loss_history.json', 'w') as f:
        json.dump(loss_history, f, indent=2)

    # Save loss history as TXT
    with open(f'{folder_name}/loss_history.txt', 'w') as f:
        f.write("Epoch\tLoss\n")
        for entry in loss_history:
            f.write(f"{entry['epoch']}\t{entry['loss']:.6f}\n")

    # Generate and save synthetic data as NPZ
    with torch.no_grad():
        # synthetic_samples = diffusion.sample(45000)
        synthetic_samples = diffusion.sample(10000)
        synthetic_samples = synthetic_samples.cpu().numpy().transpose(0, 2, 1)

    np.savez_compressed(
        f'{folder_name}/synthetic_data.npz',
        data=synthetic_samples
    )


def main_test(model_folder_path, num_samples=45000):
    """
    Load a trained diffusion model and generate synthetic data.

    Args:
        model_folder_path: Path to the folder containing the saved model and params
        num_samples: Number of synthetic samples to generate
    """
    # -------------------------
    # LOAD SAVED PARAMETERS
    # -------------------------
    params_path = os.path.join(model_folder_path, 'params.txt')
    with open(params_path, 'r') as f:
        params = json.load(f)

    # Extract parameters
    dataset = params['dataset_name']
    seq = params['seq_len']
    latent_dim = params['hidden_dim']
    n_heads = params['n_head']
    num_layers = params['num_of_layers']
    timesteps_local = params['timesteps']
    obj = params['objective']
    beta_sched = params['beta_schedule']

    print(f"Loading model with parameters:")
    print(f"  Dataset: {dataset}")
    print(f"  Sequence length: {seq}")
    print(f"  Hidden dim: {latent_dim}")
    print(f"  Timesteps: {timesteps_local}")

    # -------------------------
    # LOAD DATA (to get features dimension)
    # -------------------------
    train_data, test_data = LoadData(dataset, seq)
    train_data, test_data = np.asarray(train_data), np.asarray(test_data)
    features = train_data.shape[2]

    # -------------------------
    # SETUP DEVICE
    # -------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # -------------------------
    # RECREATE MODEL ARCHITECTURE
    # -------------------------
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
        objective=obj,
        loss_type='l2',
        beta_schedule=beta_sched
    )

    # -------------------------
    # LOAD SAVED MODEL WEIGHTS
    # -------------------------
    # Find the model file (should end with -final.pth)
    model_files = [f for f in os.listdir(model_folder_path) if f.endswith('-final.pth')]
    if not model_files:
        raise FileNotFoundError(f"No model file found in {model_folder_path}")

    model_path = os.path.join(model_folder_path, model_files[0])
    print(f"Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
    diffusion = diffusion.to(device)
    diffusion.eval()

    print(f"Model loaded successfully (trained for {checkpoint['epoch']} epochs)")

    # -------------------------
    # GENERATE SYNTHETIC DATA
    # -------------------------
    print(f"Generating {num_samples} synthetic samples...")

    with torch.no_grad():
        synthetic_samples = diffusion.sample(num_samples)
        synthetic_samples = synthetic_samples.cpu().numpy().transpose(0, 2, 1)

    print(f"Generated synthetic data shape: {synthetic_samples.shape}")

    # -------------------------
    # SAVE SYNTHETIC DATA
    # -------------------------
    timestamp = time.time()
    output_folder = os.path.join(model_folder_path, 'test_outputs')
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Save as NPZ
    npz_path = os.path.join(output_folder, f'synthetic_data_{timestamp:.0f}.npz')
    np.savez_compressed(npz_path, data=synthetic_samples)
    print(f"Saved synthetic data (NPZ): {npz_path}")

    # Save as NPY
    npy_path = os.path.join(output_folder, f'synthetic_data_{timestamp:.0f}.npy')
    np.save(npy_path, synthetic_samples)
    print(f"Saved synthetic data (NPY): {npy_path}")

    # -------------------------
    # VISUALIZE (optional)
    # -------------------------
    test_data_transposed = test_data.transpose(0, 2, 1)

    writer = SummaryWriter(log_dir=output_folder, comment='test_generation')

    visualize(
        test_data_transposed,
        synthetic_samples,
        dataset,
        seq,
        output_folder,
        epoch=checkpoint['epoch'],
        writer=writer
    )

    writer.close()
    print(f"Visualization saved to: {output_folder}")

    return synthetic_samples



if __name__ == "__main__":
    # main_train("iot")
    # model_folder = 'saved_files/1770796660.6585-custom-transformers-iot-l1-cosine-365-pred_v'
    # synthetic_data = main_test(model_folder, num_samples=10000)

    # main_train("m5")
    model_folder = 'saved_files/1771403691.0601-custom-transformers-m5-l1-cosine-365-pred_v'
    synthetic_data = main_test(model_folder, num_samples=10000)
