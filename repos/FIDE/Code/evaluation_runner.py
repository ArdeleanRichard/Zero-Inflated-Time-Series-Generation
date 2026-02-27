"""
Script to run comprehensive evaluations on trained model
"""

import torch

from constants import device
from evaluation import (evaluate_predictive_score, evaluate_discriminative_score, visualize_pca_tsne)
from general_utilities import plot_kde, KS_Test, CMD, KL_JS_divergence, CRPS
from data_process import fit_AR_model



def run_comprehensive_evaluation(real_data, samples_ddpm, seq_len, n_seq, 
                                 block_maxima_real_data_value, 
                                 bm_samples_ddpm_value,
                                 is_fit_AR=False):
    """
    Run comprehensive evaluation of generated samples
    
    Args:
        real_data: Real time series data (numpy array or torch tensor)
        samples_ddpm: Generated samples from DDPM
        seq_len: Sequence length
        n_seq: Number of features
        block_maxima_real_data_value: Real block maxima values
        bm_samples_ddpm_value: Generated block maxima values
        is_fit_AR: Whether to fit AR model
    """
    
    # Convert to numpy if needed
    if torch.is_tensor(real_data):
        real_data = real_data.cpu().numpy()
    if torch.is_tensor(samples_ddpm):
        samples_ddpm = samples_ddpm.cpu().numpy()
    
    # Ensure correct shapes
    real_data_reshaped = real_data.reshape(real_data.shape[0], seq_len, n_seq)
    samples_ddpm_reshaped = samples_ddpm[:len(real_data)].reshape(-1, seq_len, n_seq)
    
    print("="*60)
    print("COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # 1. Predictive Score Evaluation
    print("\n1. PREDICTIVE SCORE EVALUATION")
    print("-"*60)
    results = evaluate_predictive_score(real_data_reshaped, samples_ddpm_reshaped, seq_len)
    print(results)
    
    # 2. Discriminative Score Evaluation
    print("\n2. DISCRIMINATIVE SCORE EVALUATION")
    print("-"*60)
    disc_score = evaluate_discriminative_score(real_data_reshaped, samples_ddpm_reshaped, 
                                                seq_len, n_seq)
    
    # 3. PCA and t-SNE Visualization
    print("\n3. PCA AND t-SNE VISUALIZATION")
    print("-"*60)
    visualize_pca_tsne(real_data_reshaped, samples_ddpm_reshaped, seq_len)
    
    # 4. Statistical Tests on Block Maxima
    print("\n4. BLOCK MAXIMA STATISTICAL TESTS")
    print("-"*60)
    plot_kde(block_maxima_real_data_value, bm_samples_ddpm_value, 
             x_axis_label="Max Value", 
             title="KDE Density Plot of Max Values (Real vs DDPM Generated)")
    KS_Test(block_maxima_real_data_value, bm_samples_ddpm_value)
    CMD(block_maxima_real_data_value, bm_samples_ddpm_value)
    KL_JS_divergence(block_maxima_real_data_value, bm_samples_ddpm_value)
    CRPS(block_maxima_real_data_value, bm_samples_ddpm_value)
    
    # 5. Statistical Tests on All Values
    print("\n5. ALL VALUES STATISTICAL TESTS")
    print("-"*60)
    bm_samples_ddpm = samples_ddpm.reshape(-1)
    plot_kde(real_data.reshape(-1), bm_samples_ddpm, 
             x_axis_label="All Values", 
             title="KDE Density Plot of All Values (Real vs Generated (DDPM))")
    KS_Test(real_data.reshape(-1), bm_samples_ddpm)
    CMD(real_data.reshape(-1), bm_samples_ddpm)
    KL_JS_divergence(real_data.reshape(-1), bm_samples_ddpm)
    CRPS(real_data.reshape(-1), bm_samples_ddpm)
    
    # 6. AR Model Fitting (if requested)
    if is_fit_AR:
        print("\n6. AR MODEL COEFFICIENT COMPARISON")
        print("-"*60)
        fit_AR_model(
            torch.tensor(bm_samples_ddpm.reshape(-1, seq_len)).to(device),
            true_coeffs=[0.5], 
            order=1,
            seq_len=seq_len
        )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print("="*60)


if __name__ == "__main__":
    print("This is an evaluation module.")
    print("Import and use run_comprehensive_evaluation() function in your main script.")
