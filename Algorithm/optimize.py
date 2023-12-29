import joblib
import multiprocessing
import pandas as pd
import numpy as np
from typing import Iterable, Any
from itertools import product
from utils import write_to_excel
from BAR_learn import data_setup


def grid_parameters(parameters: dict[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))


parameters = {
    "model": [joblib.load('Notebooks/classification_model.joblib')],
    "X": [pd.read_pickle('Data/X_t_train.pkl')],
    "y": [pd.read_pickle('Data/y_t_train.pkl')],
    "X_s": [pd.read_parquet(path='Notebooks/turbofan_features.parquet', engine='pyarrow')],
    "learning_rate": [0.6],
    "epochs": [120],
    "q": [50],
    "bias": [1],
    "n_splits": [5],
    "punishment": [7],
    "factor": [135]}


def get_performance(parameters):
    print(parameters)
    parameters = parameters[0]
    result, weights = data_setup(**parameters)
    parameters['result'] = result
    parameters['weights'] = np.array2string(pd.DataFrame(np.concatenate(weights, axis=0)).values.flatten())
    return parameters


def parallel_grid_search(parameter_sets, num_processes):
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(get_performance, [parameter_sets]*num_processes)
    pool.close()
    pool.join()
    return results


if __name__ == "__main__":
    parameter_grid = [
        {"learning_rate": lr, "epochs": ep, "q": q, "bias": b, "n_splits": n_s, "punishment": pun, "factor": fac}
        for lr, ep, q, b, n_s, pun, fac in product(
            parameters["learning_rate"],
            parameters["epochs"],
            parameters["q"],
            parameters["bias"],
            parameters["n_splits"],
            parameters["punishment"],
            parameters["factor"]
        )
    ]

    num_processes = multiprocessing.cpu_count()
    for i in range(5):
        grid_search_results = parallel_grid_search(parameter_grid, num_processes)
        write_to_excel(grid_search_results, path='Data/Optimization_3.xlsx')
    print("Grid search results:", grid_search_results)
    print("finished")
