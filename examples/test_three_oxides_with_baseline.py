from examples.functions import create_oxide_function, create_baseline_function, create_solution
from models.three_oxides_and_baseline import create_pinn_model
from matplotlib import pyplot as plt
import numpy as np
import deepxde as dde


def create_reference_solution(sigma, solution_oxides, baseline_parameters, verbose=True):

    oxide_functions = []
    for oxide in solution_oxides:
        oxide_functions.append(create_oxide_function(oxide["K"] * 1e1, oxide["E"] * 1e5, oxide["v_initial"], oxide["t_initial"]))
    baseline_func = create_baseline_function(**baseline_parameters)

    number_of_points = 1000
    t_grid = np.linspace(0, 600, number_of_points).reshape(-1, 1).astype("float64")
    solution = create_solution(oxide_functions + [baseline_func])
    solution_values = solution(t_grid).astype("float64") + np.random.normal(0, sigma, number_of_points).reshape(-1, 1)

    # plot it
    if verbose:
        plt.plot(t_grid, solution_values)
        plt.show()

    return t_grid, solution_values


def test_inverse_problem(sigma, oxides, baseline_parameters):

    # create reference solution
    t_grid, solution_values = create_reference_solution(sigma, oxides, baseline_parameters)

    # create model
    model, external_trainable_variables, variable = create_pinn_model(t_grid, solution_values, oxides)

    # train adam
    loss_weights = [1e+5, 1e+5, 1e+5, 100, 1, 1, 1, 1, 100]
    model.compile("adam", lr=3e-4 / 2, loss_weights=loss_weights,
                  external_trainable_variables=external_trainable_variables)
    loss_history, train_state = model.train(iterations=100000, callbacks=[variable])
    dde.saveplot(loss_history, train_state, issave=True, isplot=True)

    # train lbfgs
    model.compile("L-BFGS", external_trainable_variables=external_trainable_variables)
    loss_history, train_state = model.train(callbacks=[variable])
    dde.saveplot(loss_history, train_state, issave=True, isplot=True)

    return model, train_state
