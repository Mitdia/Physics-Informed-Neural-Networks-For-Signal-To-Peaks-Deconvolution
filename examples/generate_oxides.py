import numpy as np
from examples.functions import create_oxide_function
from matplotlib import pyplot as plt


def find_t_initial(k, e, t_shift):
    t_grid = np.linspace(0, 600, 6000).reshape(-1, 1)
    for t_init in range(0, 400, 10):
        oxide_function = create_oxide_function(k * 1e1, e * 1e5, 1, t_init, t_shift)
        if np.max(oxide_function(t_grid)) < 20:
            return t_init
    return None

if __name__ == "__main__":
    for e in np.arange(0.5, 3.5, step=0.1):
        for k in  np.arange(2, 20, step=1):
            t_grid = np.linspace(0, 600, 6000).reshape(-1, 1)
            t_initial = find_t_initial(k, e, 1600)
            oxide_function = create_oxide_function(k * 1e1, e * 1e5, 0.1, t_initial)
            if t_initial is not None and t_initial != 0  and np.sum(oxide_function(t_grid)) > 1:

                print("{" + f"\"K\": {round(k, 2)}, \"E\": {round(e, 2)}, \"t_initial\": {round(t_initial, 2)}" + "},")

                number_of_points = 1000
                t_grid = np.linspace(0.0, 650, number_of_points).reshape(-1, 1).astype("float64")
                oxide = create_oxide_function(k * 1e+1, e * 1e+5, 0.1, t_initial, 1600)
                plt.plot(t_grid, oxide(t_grid), label=f"oxide {k}, {e}, {t_initial}")
                plt.xlabel("Time")
                plt.ylabel("Excretion speed")
                plt.legend()
                plt.grid()
                plt.show()