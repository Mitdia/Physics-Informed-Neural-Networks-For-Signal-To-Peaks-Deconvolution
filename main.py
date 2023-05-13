from examples.test_three_oxides_with_baseline import test_inverse_problem, create_reference_solution
from examples.functions import  create_oxide_function, create_baseline_function
import deepxde as dde
import numpy as np
from matplotlib import pyplot as plt


dde.config.set_random_seed(42)
dde.config.set_default_float("float64")

oxides = [{'K': 1.7, 'E': 0.4, 't_initial': 0, 'v_initial': 0.32874598834960655, "t_maximum": 235.5951984},
          {'K': 5.9, 'E': 1.3, 't_initial': 360, 'v_initial': 0.720433702225778, "t_maximum": 451.90063354},
          {'K': 4.0, 'E': 0.75, 't_initial': 0.0, 'v_initial': 0.6437706081012446, "t_maximum": 89.51317106}]
baseline = {'h': 2.088245218817516, 't_beg': 1622.829031294163, 't_end': 2229.070551531213}
t_shift = 1627.9269500782368
sigma = 0.1
oxide_functions = []
for oxide in oxides:
    oxide_functions.append(create_oxide_function(oxide["K"] * 1e1,
                                                 oxide["E"] * 1e5,
                                                 oxide["v_initial"],
                                                 oxide["t_initial"],
                                                 t_shift))

baseline_func = create_baseline_function(**baseline, t_shift=t_shift)

number_of_points = 1000
t_grid = np.linspace(0.0, 650, number_of_points).reshape(-1, 1).astype("float64")

for i, oxide in enumerate(oxide_functions):
    plt.plot(t_grid, oxide(t_grid), label=f"oxide: {i}")
plt.plot(t_grid, baseline_func(t_grid), label="Baseline")
t_grid, solution_values = create_reference_solution(sigma, oxides, baseline, t_shift=t_shift)
plt.plot(t_grid, solution_values, label="Solution")
plt.legend()
plt.grid()
plt.show()

for i, oxide in enumerate(oxide_functions):
    print(t_grid[oxide(t_grid).argmax()])

model, train_state, var = test_inverse_problem(sigma, oxides, baseline, t_shift, use_lbfgs=False)

for i in range(len(oxides)):
    plt.plot(train_state.X_train, train_state.y_pred_train[:, i], '.', label=f"oxide {i}")
plt.plot(train_state.X_train, train_state.y_pred_train[:, len(oxides)], '.', label=f"baseline")
plt.plot(train_state.X_train, train_state.y_pred_train[:, len(oxides) + 1], '.', label=f"summ")
plt.legend()
plt.grid()
plt.show()


oxides = [{"K": 4.0, "E": 0.9, "v_initial": 0.6, "t_initial": 300.0, "t_maximum": 452.55085028},
          {"K": 4.0, "E": 0.75, "v_initial": 0.2, "t_initial": 0.0, "t_maximum": 117.25575192},
          {"K": 3.3, "E": 0.7, "v_initial": 0.3, "t_initial": 150, "t_maximum": 294.98166055}]

baseline = {"h": 2, "t_beg": 1800, "t_end": 2000}
t_shift = 1600
sigma = 0.1
oxide_functions = []
for oxide in oxides:
    oxide_functions.append(create_oxide_function(oxide["K"] * 1e1,
                                                 oxide["E"] * 1e5,
                                                 oxide["v_initial"],
                                                 oxide["t_initial"],
                                                 t_shift))

baseline_func = create_baseline_function(**baseline, t_shift=t_shift)

number_of_points = 2000
t_grid = np.linspace(0, 650, number_of_points).reshape(-1, 1).astype("float64")


for i, oxide in enumerate(oxide_functions):
    plt.plot(t_grid, oxide(t_grid), label=f"oxide: {i}")
plt.plot(t_grid, baseline_func(t_grid), label="Baseline")
t_grid, solution_values = create_reference_solution(sigma, oxides, baseline, t_shift=t_shift)
plt.plot(t_grid, solution_values, label="Solution")
plt.legend()
plt.grid()
plt.show()

for i, oxide in enumerate(oxide_functions):
    print(t_grid[oxide(t_grid).argmax()])


model, train_state, var = test_inverse_problem(sigma, oxides, baseline, t_shift, use_lbfgs=False)

for i in range(len(oxides)):
    plt.plot(train_state.X_train, train_state.y_pred_train[:, i], '.', label=f"oxide {i}")

plt.plot(train_state.X_train, train_state.y_pred_train[:, len(oxides)], '.', label=f"baseline")
plt.plot(train_state.X_train, train_state.y_pred_train[:, len(oxides) + 1], '.', label=f"summ")
plt.legend()
plt.grid()
plt.show()
