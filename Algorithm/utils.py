import math

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import log_loss

def transfer_X(X, W):
    X_hat = X * W
    return X_hat


def loss_funct(W, X, y):
    y_true = y
    sample_weights = {
        1: 1.166667,
        2: 3.5,
        3: 0.538462
    }
    X_hat = transfer_X(X, W)
    model = load_model()
    output = model.predict_proba(X_hat)
    output_translated = map_output(output)
    loss = log_loss(y, output_translated, labels=np.array([0, 1, 2]), sample_weight=[sample_weights[y] for y in y_true])
    return loss


def load_model():
    model = joblib.load('../Notebooks/classification_model.joblib')
    return model


def map_output(y):
    y[:, 2] += y[:, 3]
    y = np.delete(y, 3, axis=1)
    y_norm = y / np.sum(y, axis=1)[:, np.newaxis]
    return y_norm


def get_U(W):
    # Generate Random Vector U
    # U_j ∈ R_d is a vector that is uniformly drawn at random from a unit Euclidean sphere
    U_j = np.random.normal(size=W.shape)
    U_j /= np.linalg.norm(U_j)

    return 10*U_j

def update_W(W, alpha, gradient, iteration):
    # linear
    #W = W + alpha * gradient
    # e-Function
    decay = 0.02
    W = W + math.exp(-decay*iteration) * alpha * gradient

    return W


def load_small_X():
    X = pd.read_parquet('../Notebooks/turbofan_features.parquet', engine='pyarrow')
    return X.iloc[:2000]


def load_small_dataset(sample_size):
    y = pd.read_parquet('../Notebooks/turbofan_RUL.parquet', engine='pyarrow')
    X = pd.read_parquet('../Notebooks/turbofan_features.parquet', engine='pyarrow')
    #y = y.sample(sample_size)
    #X = X.sample(sample_size)
    y = y.values.T
    flat_list = [item for sublist in y for item in sublist]
    flat_list = np.array(flat_list)
    return flat_list, X


def load_small_transfer_dataset(sample_size):
    y = pd.read_parquet('../Notebooks/turbofan_RUL.parquet', engine='pyarrow')
    X = pd.read_parquet('../Notebooks/turbofan_RUL.parquet', engine='pyarrow')


def rename_columns(df_A, df_B):
    column_names_A = df_A.columns.tolist()
    column_names_B = df_B.columns.tolist()

    extra_columns = list(set(column_names_A) - set(column_names_B))

    df_B.columns = column_names_A[:len(column_names_B)]

    for column_name in extra_columns:
        if column_name not in df_B.columns:
            df_B[column_name] = 0

    df_B = df_B[df_A.columns]
    return df_B


def print_progress_bar(iteration, total, length=50):
    progress = (iteration / total)
    arrow = '=' * int(length * progress)
    spaces = ' ' * (length - len(arrow))
    print(f'\r[{arrow}{spaces}] {int(progress * 100)}% \n', end='', flush=True)