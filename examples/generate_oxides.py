import numpy as np
from functions import create_oxide_function
from matplotlib import pyplot as plt


def find_t_initial(k, e):
    t_grid = np.linspace(0, 600, 6000).reshape(-1, 1)
    for t_init in range(0, 400, 10):
        oxide_function = create_oxide_function(k * 1e1, e * 1e5, 1, t_init)
        if np.max(oxide_function(t_grid)) < 20:
            return t_init
    return None


def find_k(e):
    for k in np.arange(0.1, 8.0, step=0.2):
        oxide_function = create_oxide_function(k * 1e1, e * 1e5, 0.1, 0)
        if np.all(np.isclose(oxide_function(np.array([0, 600])), np.array([0, 0]), atol=0.1)):
            return k


for e in np.arange(0.1, 1.5, 0.05):
    k = find_k(e)
    t_initial = find_t_initial(k, e)

    print("{" + f"\"K\": {round(k, 2)}, \"E\": {round(e, 2)}, \"t_initial\": {round(t_initial, 2)}" + "}")
