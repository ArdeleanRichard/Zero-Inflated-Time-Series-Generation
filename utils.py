import matplotlib.pyplot as plt


def plot_training_history(history, save_path, model_type='vae'):
    """
    Plot training and validation losses and save to save_path = full path.

    model_type='vae'  expects keys:
        train_loss, val_loss, train_recon_loss, train_kl_loss, train_sparsity_loss
    model_type='gan'  expects keys:
        d_loss, g_loss, g_recon_loss, g_sparsity_loss, g_fm_loss
    """
    if model_type == 'vae':
        _plot_vae_history(history, save_path)
    elif model_type == 'gan':
        _plot_gan_history(history, save_path)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'vae' or 'gan'.")


def _plot_vae_history(history, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('VAE Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_recon_loss'], label='Reconstruction Loss', linewidth=2)
    axes[1].plot(history['train_kl_loss'], label='KL Divergence Loss', linewidth=2)
    axes[1].plot(history['train_sparsity_loss'], label='Sparsity Loss', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('VAE Training Loss Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'VAE training history saved to {save_path}')


def _plot_gan_history(history, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(history['d_loss'], label='Discriminator Loss', linewidth=2)
    axes[0].plot(history['g_loss'], label='Generator Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('GAN Discriminator and Generator Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['g_recon_loss'], label='Reconstruction Loss', linewidth=2)
    axes[1].plot(history['g_sparsity_loss'], label='Sparsity Loss', linewidth=2)
    axes[1].plot(history['g_fm_loss'], label='Feature Matching Loss', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('GAN Generator Loss Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'GAN training history saved to {save_path}')


def plot_sample_comparisons(ori_samples, gen_samples, num_samples=5, save_path='sample_comparison.png'):
    """Plot comparison between real and synthetic samples. save_path = full path."""
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 3 * num_samples))

    for i in range(num_samples):
        axes[i, 0].plot(ori_samples[i], linewidth=1, alpha=0.7)
        axes[i, 0].set_title(f'Real Sample {i + 1}')
        axes[i, 0].set_ylabel('Seconds')
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(gen_samples[i], linewidth=1, alpha=0.7, color='orange')
        axes[i, 1].set_title(f'Synthetic Sample {i + 1}')
        axes[i, 1].set_ylabel('Seconds')
        axes[i, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Day')
    axes[-1, 1].set_xlabel('Day')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Sample comparison saved to {save_path}')