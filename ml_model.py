import numpy as np
import scipy.signal
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import pearsonr
from scipy.signal import correlate, detrend
import joblib
import matplotlib.pyplot as plt
import os
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


# Preprocessing functions
def notch_filter(data, fs, freq=60, Q=30):
    nyq = 0.5 * fs
    freqs = np.arange(freq, 181, freq)
    for f in freqs:
        b, a = scipy.signal.iirnotch(f / nyq, Q)
        data = scipy.signal.filtfilt(b, a, data, axis=0)
    return data


def bandpass_filter(data, fs, low=5, high=180):
    nyq = 0.5 * fs
    b, a = scipy.signal.butter(4, [low / nyq, high / nyq], btype='bandpass')
    return scipy.signal.filtfilt(b, a, data, axis=0)


def estimate_lag(ecog_data, finger_data, fs=1000, max_lag_ms=200, default_lag_ms=40):
    lmp = np.mean(ecog_data, axis=1)
    lmp = detrend(lmp)
    finger_data = detrend(finger_data)
    lmp = (lmp - np.mean(lmp)) / np.std(lmp)
    finger_data = (finger_data - np.mean(finger_data)) / np.std(finger_data)
    max_lag_samples = int(max_lag_ms * fs / 1000)
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    corr = correlate(lmp, finger_data, mode='full')[len(lmp) - max_lag_samples - 1: len(lmp) + max_lag_samples]

    # Find the lag with maximum correlation
    optimal_lag = lags[np.argmax(corr)]
    lag_ms = optimal_lag / fs * 1000

    # If_lag is 0, use a default lag
    if optimal_lag == 0:
        print("Warning: Computed lag is 0. Using default lag of 40 ms.")
        optimal_lag = int(default_lag_ms * fs / 1000)
        lag_ms = default_lag_ms

    print(f"Optimal lag: {optimal_lag} samples ({lag_ms:.0f} ms)")
    print(f"Max correlation value: {np.max(corr):.4f}")
    return optimal_lag, lag_ms


def align_data(ecog_data, finger_data, lag_samples):
    if lag_samples == 0:
        return ecog_data, finger_data  # No alignment needed
    elif lag_samples > 0:
        ecog_aligned = ecog_data[lag_samples:]
        finger_aligned = finger_data[:-lag_samples]
    else:
        lag_samples = abs(lag_samples)
        ecog_aligned = ecog_data[:-lag_samples]
        finger_aligned = finger_data[lag_samples:]

    # Ensure arrays are not empty
    if ecog_aligned.shape[0] == 0 or finger_aligned.shape[0] == 0:
        raise ValueError(f"Alignment resulted in empty array. Lag samples: {lag_samples}, "
                         f"ECoG shape: {ecog_data.shape}, Finger shape: {finger_data.shape}")
    return ecog_aligned, finger_aligned


def sliding_window_features(ecog_data, fs=1000, window_ms=100, overlap_ms=50):
    n_samples, n_channels = ecog_data.shape
    window_size = int(window_ms * fs / 1000)
    step_size = int((window_ms - overlap_ms) * fs / 1000)
    n_windows = (n_samples - window_size) // step_size + 1

    starts = np.arange(n_windows) * step_size
    window_indices = starts[:, None] + np.arange(window_size)
    windows = ecog_data[window_indices]

    f, Pxx = scipy.signal.welch(windows, fs=fs, nperseg=window_size, axis=1)

    bands = [(8, 12), (18, 24), (35, 42), (42, 70), (70, 100), (100, 140), (140, 180)]
    n_bands = len(bands)

    band_masks = np.zeros((n_bands, len(f)), dtype=bool)
    for j, (low, high) in enumerate(bands):
        band_masks[j] = (f >= low) & (f <= high)

    band_powers = np.zeros((n_windows, n_bands, n_channels))
    for j in range(n_bands):
        if np.any(band_masks[j]):
            band_powers[:, j, :] = np.mean(Pxx[:, band_masks[j], :], axis=1)

    band_powers = band_powers.reshape(n_windows, n_bands * n_channels)
    lmp = np.mean(windows, axis=1)
    features = np.hstack((band_powers, lmp))
    features = np.nan_to_num(features, nan=0.0)
    return features

#
# def train_xgboost_model(data_path, output_dir='models_try', fs=1000):
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Check GPU support
#     try:
#         params = xgb.XGBRegressor(tree_method='hist', device='cuda').get_params()
#         print("XGBoost GPU support is available!")
#         gpu_params = {'tree_method': 'hist', 'device': 'cuda'}
#     except Exception as e:
#         print(f"XGBoost GPU support not available, falling back to CPU: {e}")
#         gpu_params = {}
#
#     data = loadmat(data_path)
#     train_ecog = data['train_ecog']
#     train_dg = data['train_dg']
#
#     for patient_idx in range(3):
#         print(f"\nProcessing Patient {patient_idx + 1}")
#         ecog_data = train_ecog[patient_idx, 0]
#         dg_data = train_dg[patient_idx, 0]
#
#         # Data alignment and preprocessing
#         finger_data = dg_data[:, 0]
#         lag_samples, lag_ms = estimate_lag(ecog_data, finger_data, fs)
#         ecog_aligned, dg_aligned = align_data(ecog_data, dg_data, lag_samples)
#
#         # Data augmentation - add small noise variations
#         noise = np.random.normal(0, 0.001, ecog_aligned.shape)
#         ecog_augmented = ecog_aligned + noise
#
#         # Filtering pipeline
#         ecog_notched = notch_filter(ecog_augmented, fs)
#         ecog_filtered = bandpass_filter(ecog_notched, fs)
#         ecog_smoothed = scipy.signal.savgol_filter(ecog_filtered, 101, 3, axis=0)
#
#         # Feature extraction
#         features = sliding_window_features(ecog_smoothed, fs)
#
#         # Prepare labels with proper alignment
#         window_ms, overlap_ms = 100, 50
#         window_size = int(window_ms * fs / 1000)
#         step_size = int((window_ms - overlap_ms) * fs / 1000)
#         n_windows = features.shape[0]
#         label_indices = np.arange(window_size // 2, dg_aligned.shape[0], step_size)
#         label_indices = label_indices[:n_windows]
#         labels = dg_aligned[label_indices]
#
#         # Feature scaling
#         scaler = StandardScaler()
#         features_scaled = scaler.fit_transform(features)
#         joblib.dump(scaler, os.path.join(output_dir, f'patient_{patient_idx + 1}_scaler.pkl'))
#
#         n_kinematic_params = labels.shape[1]
#         correlations = []
#         mses = []
#
#         # Time-series aware cross-validation
#         tscv = TimeSeriesSplit(n_splits=5)
#
#         for param_idx in range(n_kinematic_params):
#             print(f"\nTraining XGBoost for kinematic parameter {param_idx + 1}")
#
#             # Base model with regularization
#             xgb_model = xgb.XGBRegressor(
#                 objective='reg:squarederror',
#                 random_state=42,
#                 **gpu_params,
#                 reg_alpha=1,  # L1 regularization
#                 reg_lambda=2.0,  # L2 regularization
#                 gamma=0.3,  # Minimum loss reduction
#                 max_delta_step=0.1,  # Helps prevent overfitting
#                 subsample=0.8,  # Random row sampling
#                 colsample_bytree=0.8  # Random column sampling
#             )
#
#             # Conservative parameter grid
#             param_grid = {
#                 'n_estimators': [50, 100],  # Reduced complexity
#                 'max_depth': [3, 4],  # Shallower trees
#                 'learning_rate': [0.01, 0.05],  # Smaller learning rates
#                 'min_child_weight': [1, 3]  # Controls splits
#             }
#
#             # Setup grid search with early stopping
#             grid_search = GridSearchCV(
#                 estimator=xgb_model,
#                 param_grid=param_grid,
#                 cv=tscv,  # Time-series cross-validation
#                 scoring='neg_mean_squared_error',
#                 n_jobs=1,
#                 verbose=1
#             )
#
#             # Fit with early stopping
#             eval_set = [(features_scaled, labels[:, param_idx])]
#             grid_search.fit(
#                 features_scaled,
#                 labels[:, param_idx],
#                 eval_set=eval_set,
#                 verbose=False
#             )
#
#             best_xgb = grid_search.best_estimator_
#             print(f"Best parameters: {grid_search.best_params_}")
#
#             # Save model
#             model_path = os.path.join(output_dir, f'patient_{patient_idx + 1}_param_{param_idx + 1}_xgb.pkl')
#             joblib.dump(best_xgb, model_path)
#
#             # Feature importance analysis
#             importances = best_xgb.feature_importances_
#             important_features = np.where(importances > np.mean(importances))[0]
#             print(f"Number of important features: {len(important_features)}/{features.shape[1]}")
#
#             # Evaluation
#             y_pred = best_xgb.predict(features_scaled)
#             mse = mean_squared_error(labels[:, param_idx], y_pred)
#             corr, _ = pearsonr(labels[:, param_idx], y_pred)
#             correlations.append(corr)
#             mses.append(mse)
#             print(f"Parameter {param_idx + 1} - MSE: {mse:.4f}, Correlation: {corr:.4f}")
#
#             # Plotting first 200 samples
#             plt.figure(figsize=(10, 4))
#             plt.plot(labels[:200, param_idx], 'b-', label='Actual')
#             plt.plot(y_pred[:200], 'r--', label='Predicted')
#             plt.title(f"Patient {patient_idx + 1}, Parameter {param_idx + 1}")
#             plt.legend()
#             plt.savefig(os.path.join(output_dir, f'patient_{patient_idx + 1}_param_{param_idx + 1}_plot.png'))
#             plt.close()
#
#         print(f"\nPatient {patient_idx + 1} Summary:")
#         print(f"Mean Correlation: {np.mean(correlations):.4f}")
#         print(f"Mean MSE: {np.mean(mses):.4f}")
#
#
# if __name__ == "__main__":
#     data_path = "raw_training_data.mat"
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"Data file not found at {data_path}")
#
#     train_xgboost_model(data_path)

def train_xgboost_model(data_path, output_dir='models_bayes_opt', fs=1000):
    os.makedirs(output_dir, exist_ok=True)

    # Check GPU support
    try:
        gpu_params = {'tree_method': 'hist', 'device': 'cuda'}
        print("Using GPU acceleration")
    except:
        gpu_params = {}
        print("Using CPU")

    data = loadmat(data_path)
    train_ecog = data['train_ecog']
    train_dg = data['train_dg']

    for patient_idx in range(3):
        print(f"\n=== Processing Patient {patient_idx + 1} ===")
        ecog_data = train_ecog[patient_idx, 0]
        dg_data = train_dg[patient_idx, 0]

        # 90:10 temporal split
        split_idx = int(len(ecog_data) * 0.9)
        ecog_train, ecog_test = ecog_data[:split_idx], ecog_data[split_idx:]
        dg_train, dg_test = dg_data[:split_idx], dg_data[split_idx:]

        # Data processing pipeline - TRAIN SET
        finger_train = dg_train[:, 0]
        lag_samples, lag_ms = estimate_lag(ecog_train, finger_train, fs)
        ecog_aligned, dg_aligned = align_data(ecog_train, dg_train, lag_samples)

        # Data augmentation
        noise = np.random.normal(0, 0.001, ecog_aligned.shape)
        ecog_augmented = ecog_aligned + noise

        # Filtering
        ecog_notched = notch_filter(ecog_augmented, fs)
        ecog_filtered = bandpass_filter(ecog_notched, fs)
        ecog_smoothed = scipy.signal.savgol_filter(ecog_filtered, 101, 3, axis=0)

        # Feature extraction
        features_train = sliding_window_features(ecog_smoothed, fs)

        # Prepare labels
        window_ms, overlap_ms = 100, 50
        window_size = int(window_ms * fs / 1000)
        step_size = int((window_ms - overlap_ms) * fs / 1000)
        n_windows = features_train.shape[0]
        label_indices = np.arange(window_size // 2, dg_aligned.shape[0], step_size)
        label_indices = label_indices[:n_windows]
        labels_train = dg_aligned[label_indices]

        # Process TEST SET with same transformations
        ecog_test_aligned = ecog_test[lag_samples:] if lag_samples > 0 else ecog_test
        dg_test_aligned = dg_test[:-lag_samples] if lag_samples > 0 else dg_test
        ecog_test_notched = notch_filter(ecog_test_aligned, fs)
        ecog_test_filtered = bandpass_filter(ecog_test_notched, fs)
        ecog_test_smoothed = scipy.signal.savgol_filter(ecog_test_filtered, 101, 3, axis=0)
        features_test = sliding_window_features(ecog_test_smoothed, fs)
        n_windows_test = features_test.shape[0]
        test_label_indices = np.arange(window_size // 2, dg_test_aligned.shape[0], step_size)
        test_label_indices = test_label_indices[:n_windows_test]
        labels_test = dg_test_aligned[test_label_indices]

        # Feature scaling
        scaler = StandardScaler()
        features_train_scaled = scaler.fit_transform(features_train)
        features_test_scaled = scaler.transform(features_test)
        joblib.dump(scaler, os.path.join(output_dir, f'patient_{patient_idx + 1}_scaler.pkl'))

        # Fixed parameters
        fixed_params = {
            'objective': 'reg:squarederror',
            'random_state': 42,
            **gpu_params,
            'eval_metric': 'rmse',
            'early_stopping_rounds': 20,
            'monotone_constraints': (0, 0, 0, 0, 0)  # No enforced trends
        }

        # Bayesian optimization search space
        search_spaces = {
            'learning_rate': Real(0.01, 0.1, 'log-uniform'),
            'max_depth': Integer(2, 6),
            'n_estimators': Integer(50, 200),
            'reg_alpha': Real(0.5, 2),
            'reg_lambda': Real(1, 3),
            'gamma': Real(0, 0.3),
            'subsample': Real(0.6, 0.9),
            'colsample_bytree': Real(0.6, 0.9),
            'min_child_weight': Integer(1, 5),
            'max_delta_step': Real(0, 0.3)
        }

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=4)

        for param_idx in range(labels_train.shape[1]):
            print(f"\n>> Optimizing Parameter {param_idx + 1}")

            opt = BayesSearchCV(
                estimator=xgb.XGBRegressor(**fixed_params),
                search_spaces=search_spaces,
                n_iter=32,  # Number of optimization steps
                cv=tscv,
                scoring='neg_mean_squared_error',
                verbose=1,
                n_jobs=1
            )

            opt.fit(
                features_train_scaled,
                labels_train[:, param_idx],
                eval_set=[(features_train_scaled, labels_train[:, param_idx]),
                          (features_test_scaled, labels_test[:, param_idx])]
            )

            best_model = opt.best_estimator_
            print(f"\nBest parameters: {opt.best_params_}")

            # Evaluation
            train_pred = best_model.predict(features_train_scaled)
            test_pred = best_model.predict(features_test_scaled)

            train_corr = pearsonr(labels_train[:, param_idx], train_pred)[0]
            test_corr = pearsonr(labels_test[:, param_idx], test_pred)[0]
            train_mse = mean_squared_error(labels_train[:, param_idx], train_pred)
            test_mse = mean_squared_error(labels_test[:, param_idx], test_pred)

            print(f"TRAIN - Corr: {train_corr:.4f}, MSE: {train_mse:.4f}")
            print(f"TEST  - Corr: {test_corr:.4f}, MSE: {test_mse:.4f}")
            print(f"Generalization Gap: {abs(train_corr - test_corr):.4f}")

            # Save model
            model_path = os.path.join(output_dir,
                                      f'patient_{patient_idx + 1}_param_{param_idx + 1}_xgb.pkl')
            joblib.dump(best_model, model_path)

            # Plot test set predictions
            plt.figure(figsize=(12, 5))
            plt.plot(labels_test[:500, param_idx], 'b-', label='Actual', alpha=0.7)
            plt.plot(test_pred[:500], 'r-', label='Predicted', alpha=0.7)
            plt.title(f"Patient {patient_idx + 1} Parameter {param_idx + 1}\n"
                      f"Test Corr: {test_corr:.3f}, MSE: {test_mse:.3f}")
            plt.legend()
            plt.savefig(os.path.join(output_dir,
                                     f'patient_{patient_idx + 1}_param_{param_idx + 1}_test_plot.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()


if __name__ == "__main__":
    data_path = "raw_training_data.mat"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    train_xgboost_model(data_path)