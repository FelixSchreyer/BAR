import joblib
import multiprocessing
import pandas as pd
from typing import Iterable, Any
from itertools import product
from Algorithm.utils import write_to_excel
from BAR_learn import data_setup


def grid_parameters(parameters: dict[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))


parameters = {
    "model": [joblib.load('../Notebooks/classification_model.joblib')],
    "X": [pd.read_pickle('../Data/X_t_train.pkl')],
    "y": [pd.read_pickle('../Data/y_t_train.pkl')],
    "X_s": [pd.read_parquet(path='../Notebooks/turbofan_features.parquet', engine='pyarrow')],
    "learning_rate": [round(0.1 + i * 0.1, 1) for i in range(int(1 / 0.1) + 1)],
    "epochs": [25, 50, 75, 100],
    "q": [5, 10, 15, 20],
    "bias": [1],
    "n_splits": [1, 2, 3, 4, 5, 10],
    "punishment": [0, 0.5, 1, 1.5, 2, 5]}


def get_performance(parameters):
    print(parameters)
    result = data_setup(**parameters)
    parameters['result'] = result
    return parameters


def parallel_grid_search(parameter_sets, num_processes):
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(get_performance, parameter_sets)
    pool.close()
    pool.join()
    return results


if __name__ == "__main__":
    parameter_grid = [
        {"learning_rate": lr, "epochs": ep, "q": q, "bias": b, "n_splits": n_s, "punishment": pun}
        for lr, ep, q, b, n_s, pun in product(
            parameters["learning_rate"],
            parameters["epochs"],
            parameters["q"],
            parameters["bias"],
            parameters["n_splits"],
            parameters["punishment"]
        )
    ]

    num_processes = multiprocessing.cpu_count()

    grid_search_results = parallel_grid_search(parameter_grid, num_processes)
    write_to_excel(grid_search_results, path='../Data/Optimization.xlsx')
    print("Grid search results:", grid_search_results)
    print("finished")
