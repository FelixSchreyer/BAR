import pandas
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from utils import *


def one_sided_averaged_gradient_estimator(W, q, beta, bias, X, y, iteration, learning_rate, punishment, num_processes, factor):
    """
    Creates Updated Weights based on Zeroth Order-Optimisation.
    Calculates the gradient of W in q different directions based on a loss function

    :param iteration: Iteration in loop
    :type iteration: int
    :param W: Weights used for Translation
    :type W: numpy.ndarray
    :param q: Number of Gradients created
    :type q: int
    :param beta: Learning Rate
    :type beta: float
    :param bias: scalar balancing bias and variance trade-off of the estimator
    :type bias: int
    :param X: Features
    :type X: pandas.DataFrame
    :param y: Classes
    :type y: pandas.DataFrame
    :return: Updated Weights
    :rtype: numpy.ndarray
    """

    gradient_estimates = []

    for _ in range(q):
        U_j = get_U(W, factor)

        loss_plus = loss_funct(W=W + beta * U_j, X=X, y=y, punishment=punishment)
        loss_minus = loss_funct(W=W, X=X, y=y, punishment=punishment)
        g_j = bias / beta * (loss_minus - loss_plus) * U_j

        gradient_estimates.append(g_j)
    averaged_gradient = np.mean(gradient_estimates, axis=0)

    W_updated = update_W(W=W, alpha=learning_rate, gradient=averaged_gradient, iteration=iteration)
    return W_updated


def data_setup(
               X=pd.read_pickle('Data/X_t_train.pkl'),
               y=pd.read_pickle('Data/y_t_train.pkl'),
               X_s=pd.read_parquet(path='Notebooks/turbofan_features.parquet', engine='pyarrow'),
               learning_rate=0.4, epochs=20, q=20, beta=1, bias=1, n_splits=5, punishment=0.5, num_processes=5, factor=50):
    loss_hist = []
    weights = []

    # Rename Columns to Target Column Names
    X = rename_columns(target=X_s, source=X)

    # Scale Dataframe
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Create Weights
    num_rows, num_cols = X.shape
    #W = np.random.rand(1, num_cols)
    W = np.ones((1, num_cols))
    #W = np.full(num_cols, -2)

    #W = pd.read_pickle('../Notebooks/weights_3.pkl')
    W_0 = W

    # Check initial performance
    initial_perf = loss_funct(W, X, y, punishment=punishment)
    print("Initial performance is: ", initial_perf)
    loss_hist.append(initial_perf)
    weights.append(W)

    oversampler = RandomOverSampler(sampling_strategy='auto')
    undersampler = RandomUnderSampler(sampling_strategy='auto')


    for config in range(n_splits):
        if config % 2 == 1:
            X_train_resampled, y_train_resampled = oversampler.fit_resample(X, y)
        else:
            X_train_resampled, y_train_resampled = undersampler.fit_resample(X, y)


        # Update weights epoch times
        for _ in range(epochs):
            W = one_sided_averaged_gradient_estimator(W=W, q=q, beta=beta, bias=bias, X=X_train_resampled,
                                                      y=y_train_resampled, iteration=_,
                                                      learning_rate=learning_rate, punishment=punishment, num_processes=num_processes, factor=factor)
            loss_hist.append(loss_funct(W, X, y, punishment=punishment))
            weights.append(W)

        print_progress_bar(config + 1, n_splits)

    # Check final performance
    print("Best W is ", weights[loss_hist.index(min(loss_hist))])
    print("Best performance is: ", min(loss_hist))
    print("Corresponding performance is ", loss_funct(weights[loss_hist.index(min(loss_hist))], X, y, 2))

    # Print performance over iterations
    #plot_loss(loss_hist=loss_hist)

    store_weights(file_path="weights.pkl", weights=weights[loss_hist.index(min(loss_hist))], loss=min(loss_hist), W=W, W_0=W_0)

    result = calculate_f1_score(W, X, y)
    return result




if __name__ == '__main__':
    num_processes = multiprocessing.cpu_count()

    result = data_setup(
               X=pd.read_pickle('Data/X_t_train.pkl'),
               y=pd.read_pickle('Data/y_t_train.pkl'),
               X_s=pd.read_parquet(path='Notebooks/turbofan_features.parquet', engine='pyarrow'),
               learning_rate=1, epochs=120, q=15, beta=1, bias=1, n_splits=5, punishment=5, num_processes=1, factor=100)
