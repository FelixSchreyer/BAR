import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from utils import *

# Load Model
model = joblib.load('../Notebooks/classification_model.joblib')
# Load Data
X = pd.read_pickle('../Data.nosync+/X_t_train.pkl')
y = pd.read_pickle('../Data.nosync+/y_t_train.pkl')
X_s = pd.read_parquet('../Notebooks/turbofan_features.parquet', engine='pyarrow')
# Define Parameters
learning_rate = 0.1
epochs = 60
q = 10
beta = 1
bias = 1


def one_sided_averaged_gradient_estimator(W, q, beta, bias, X, y, iteration):
    """
    Creates Updated Weights based on Zeroth Order-Optimisation.
    Calculates the gradient of W in q different directions based on a loss function

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
        U_j = get_U(W)

        loss_plus = loss_funct(W=W + beta * U_j, X=X, y=y)
        loss_minus = loss_funct(W=W, X=X, y=y)
        g_j = bias / beta * (loss_minus - loss_plus) * U_j

        gradient_estimates.append(g_j)

    averaged_gradient = np.mean(gradient_estimates, axis=0)

    W_updated = update_W(W=W, alpha=learning_rate, gradient=averaged_gradient, iteration=iteration)

    return W_updated


if __name__ == '__main__':

    loss_hist = []
    weights = []
    losses = []

    # Prep X by renaming columns and scaling data
    X = rename_columns(X_s, X)
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Create Random Weights
    num_rows, num_cols = X.shape
    W = np.random.rand(1, num_cols)

    # Check initial performance
    print("Initial performance is: ", loss_funct(W, X, y))

    n_splits = 5

    oversampler = RandomOverSampler(sampling_strategy='auto')
    undersampler = RandomUnderSampler(sampling_strategy='auto')

    for config in range(n_splits):
        if config % 2 == 1:
            X_train_resampled, y_train_resampled = oversampler.fit_resample(X, y)
        else:
            X_train_resampled, y_train_resampled = oversampler.fit_resample(X, y)

        # Update weights epoch times
        for _ in range(epochs):
            W = one_sided_averaged_gradient_estimator(W=W, q=q, beta=beta, bias=bias, X=X, y=y, iteration=epochs)
            loss_hist.append(loss_funct(W, X * W, y))
            weights.append(W)

        print_progress_bar(config + 1, n_splits)


    # Check final performance
    print("best W is ", weights[loss_hist.index(min(loss_hist))])
    print("Final performance is: ", min(loss_hist))

    # while not loss_queue.empty():
    #    loss_hist.append(loss_queue.get())

    # Print performance over iterations
    plt.bar(range(len(loss_hist)), loss_hist)

    # Add labels and a title
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.title('Bar Plot of Floats')

    # Show the plot
    plt.show()

    file_path = "weights.pkl"

    # Save the list as a binary file
    with open(file_path, "wb") as file:
        pickle.dump(weights[loss_hist.index(min(loss_hist))], file)

    print("List saved to", file_path)
