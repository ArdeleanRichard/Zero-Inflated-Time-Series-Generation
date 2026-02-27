"""
Evaluation utilities for time series generation models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError

# Set matplotlib parameters
from pylab import rcParams
rcParams['figure.figsize'] = 24, 10
plt.style.use('default')
plt.rcParams.update({'font.size': 14})

def RNN_regression(units):
    """
    Create a simple RNN model for regression
    
    Args:
        units: Number of GRU units
        
    Returns:
        Compiled Keras model
    """
    opt = Adam(name='AdamOpt')
    loss = MeanAbsoluteError(name='MAE')
    model = Sequential()
    model.add(GRU(units=units, name=f'RNN_1'))
    model.add(Dense(units=1, activation='sigmoid', name='OUT'))
    model.compile(optimizer=opt, loss=loss)
    return model


def evaluate_predictive_score(real_data, synth_data, seq_len, train_test_split=0.9):
    """
    Evaluate predictive score using RNN regression
    
    Args:
        real_data: Real time series data
        synth_data: Synthetic time series data
        seq_len: Sequence length
        train_test_split: Train/test split ratio
        
    Returns:
        results: DataFrame with evaluation metrics
    """
    n_events = len(real_data)
    
    # Split data on train and test
    idx = np.arange(n_events)
    n_train = int(train_test_split * n_events)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    
    # Define the X for synthetic and real data
    X_real_train = real_data[train_idx, :seq_len-1, :]
    X_synth_train = synth_data[train_idx, :seq_len-1, :]
    
    X_real_test = real_data[test_idx, :seq_len-1, :]
    y_real_test = real_data[test_idx, -1, :]
    
    # Define the y for synthetic and real datasets
    y_real_train = real_data[train_idx, -1, :]
    y_synth_train = synth_data[train_idx, -1, :]
    
    print('Synthetic X train: {}'.format(X_synth_train.shape))
    print('Real X train: {}'.format(X_real_train.shape))
    print('Synthetic y train: {}'.format(y_synth_train.shape))
    print('Real y train: {}'.format(y_real_train.shape))
    print('Real X test: {}'.format(X_real_test.shape))
    print('Real y test: {}'.format(y_real_test.shape))
    
    # Training the model with the real train data
    ts_real = RNN_regression(12)
    early_stopping = EarlyStopping(monitor='val_loss')
    
    real_train = ts_real.fit(x=X_real_train,
                              y=y_real_train,
                              validation_data=(X_real_test, y_real_test),
                              epochs=200,
                              batch_size=128,
                              callbacks=[early_stopping],
                              verbose=0)
    
    # Training the model with the synthetic data
    ts_synth = RNN_regression(12)
    synth_train = ts_synth.fit(x=X_synth_train,
                                y=y_synth_train,
                                validation_data=(X_real_test, y_real_test),
                                epochs=200,
                                batch_size=128,
                                callbacks=[early_stopping],
                                verbose=0)
    
    # Calculate metrics
    real_predictions = ts_real.predict(X_real_test, verbose=0)
    synth_predictions = ts_synth.predict(X_real_test, verbose=0)
    
    metrics_dict = {
        'r2': [r2_score(y_real_test, real_predictions),
               r2_score(y_real_test, synth_predictions)],
        'MAE': [mean_absolute_error(y_real_test, real_predictions),
                mean_absolute_error(y_real_test, synth_predictions)]
    }
    
    results = pd.DataFrame(metrics_dict, index=['Trained with Real', 'Trained with Synthetic'])
    
    print(f"\nPredictive Score (MAE): {mean_absolute_error(y_real_test, synth_predictions)}\n")
    
    return results


def evaluate_discriminative_score(real_data, synth_data, seq_len, n_seq, train_test_split=0.9):
    """
    Evaluate discriminative score using LSTM classifier
    
    Args:
        real_data: Real time series data
        synth_data: Synthetic time series data
        seq_len: Sequence length
        n_seq: Number of features
        train_test_split: Train/test split ratio
        
    Returns:
        Discriminative score
    """
    n_series = real_data.shape[0]
    idx = np.arange(n_series)
    n_train = int(train_test_split * n_series)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    
    train_data = np.vstack((real_data[train_idx], synth_data[train_idx]))
    test_data = np.vstack((real_data[test_idx], synth_data[test_idx]))
    
    n_train_samples, n_test_samples = len(train_idx), len(test_idx)
    train_labels = np.concatenate((np.ones(n_train_samples), np.zeros(n_train_samples)))
    test_labels = np.concatenate((np.ones(n_test_samples), np.zeros(n_test_samples)))
    
    ts_classifier = Sequential([
        LSTM(2, input_shape=(seq_len, n_seq), name='LSTM'),
        Dense(1, activation='sigmoid', name='OUT')
    ], name='Time_Series_Classifier')
    
    ts_classifier.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['AUC', 'accuracy']
    )
    
    early_stopping = EarlyStopping(monitor='val_loss')
    
    result = ts_classifier.fit(
        x=train_data,
        y=train_labels,
        validation_data=(test_data, test_labels),
        epochs=100,
        batch_size=128,
        callbacks=[early_stopping],
        verbose=0
    )
    
    evaluations = ts_classifier.evaluate(x=test_data, y=test_labels, verbose=0)
    disc_score = np.abs(0.5 - evaluations[2])
    print(f"Discriminative Score: {disc_score}")
    
    return disc_score


def visualize_pca_tsne(real_data, synthetic_data, seq_len, sample_size=250):
    """
    Visualize data using PCA and t-SNE
    
    Args:
        real_data: Real time series data
        synthetic_data: Synthetic time series data
        seq_len: Sequence length
        sample_size: Number of samples to visualize
    """
    idx = np.random.permutation(real_data.shape[0])[:sample_size]
    
    real_sample = real_data[idx]
    synthetic_sample = synthetic_data[idx]
    
    synth_data_reduced = real_sample.reshape(-1, seq_len)
    real_data_reduced = np.asarray(synthetic_sample).reshape(-1, seq_len)
    
    n_components = 2
    pca = PCA(n_components=n_components)
    tsne = TSNE(n_components=n_components, n_iter=300)
    
    # The fit of the methods must be done only using the real sequential data
    pca.fit(real_data_reduced)
    
    pca_real = pd.DataFrame(pca.transform(real_data_reduced))
    pca_synth = pd.DataFrame(pca.transform(synth_data_reduced))
    
    data_reduced = np.concatenate((real_data_reduced, synth_data_reduced), axis=0)
    tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))
    
    # The scatter plots for PCA and TSNE methods
    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    
    # PCA scatter plot
    ax = fig.add_subplot(spec[0, 0])
    ax.set_title('PCA results', fontsize=20, color='red', pad=10)
    plt.scatter(pca_real.iloc[:, 0].values, pca_real.iloc[:, 1].values,
                c='black', alpha=0.2, label='Original')
    plt.scatter(pca_synth.iloc[:, 0], pca_synth.iloc[:, 1],
                c='red', alpha=0.2, label='Synthetic')
    ax.legend()
    
    # TSNE scatter plot
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.set_title('TSNE results', fontsize=20, color='red', pad=10)
    plt.scatter(tsne_results.iloc[:sample_size, 0].values, tsne_results.iloc[:sample_size, 1].values,
                c='blue', alpha=0.3, label='Original')
    plt.scatter(tsne_results.iloc[sample_size:, 0], tsne_results.iloc[sample_size:, 1],
                c='red', alpha=0.3, label='Synthetic')
    ax2.legend()
    
    fig.suptitle('Validating synthetic vs real data diversity and distributions', fontsize=16, color='grey')
    plt.show()
