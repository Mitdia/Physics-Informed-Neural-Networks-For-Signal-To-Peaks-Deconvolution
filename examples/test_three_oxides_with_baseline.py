from examples.functions import create_oxide_function, create_baseline_function, create_solution
from models.three_oxides_and_baseline import create_pinn_model
from matplotlib import pyplot as plt
import numpy as np
import deepxde as dde


def create_reference_solution(sigma,
                              k_1=4.0, e_1=0.9, v_01_true=0.6, t_01=300.0,
                              k_2=4.0, e_2=0.75, v_02_true=0.2, t_02=0.0,
                              k_3=3.3, e_3=0.7, v_03_true=0.3, t_03=150, verbose=True):

    func1 = create_oxide_function(k_1 * 10, e_1 * 1e+5, v_01_true, t_01)
    func2 = create_oxide_function(k_2 * 10, e_2 * 1e+5, v_02_true, t_02)
    func3 = create_oxide_function(k_3 * 10, e_3 * 1e+5, v_03_true, t_03)
    baseline_func = create_baseline_function(2, 1800, 2000)

    number_of_points = 1000
    t_grid = np.linspace(0, 600, number_of_points).reshape(-1, 1).astype("float64")
    solution = create_solution([func1, func2, func3, baseline_func])
    solution_values = solution(t_grid).astype("float64") + np.random.normal(0, sigma, number_of_points).reshape(-1, 1)

    # plot it
    if verbose:
        plt.plot(t_grid, solution_values)
        plt.show()

    return t_grid, solution_values


def test_inverse_problem(sigma, params):

    # create reference solution
    t_grid, solution_values = create_reference_solution(sigma, **params)

    # create model
    oxides = [{"E": params["e_1"], "K": params["k_1"], "t_initial": params["t_01"]},
              {"E": params["e_2"], "K": params["k_2"], "t_initial": params["t_02"]},
              {"E": params["e_3"], "K": params["k_3"], "t_initial": params["t_03"]}
              ]

    model, external_trainable_variables, variable = create_pinn_model(t_grid, solution_values, oxides)

    # train adam
    loss_weights = [1e+5, 1e+5, 1e+5, 100, 1, 1, 1, 1, 100]
    model.compile("adam", lr=3e-4 / 2, loss_weights=loss_weights,
                  external_trainable_variables=external_trainable_variables)
    loss_history, train_state = model.train(iterations=50000, callbacks=[variable])
    dde.saveplot(loss_history, train_state, issave=True, isplot=True)

    # train lbfgs
    model.compile("L-BFGS", external_trainable_variables=external_trainable_variables)
    loss_history, train_state = model.train(callbacks=[variable])
    dde.saveplot(loss_history, train_state, issave=True, isplot=True)

    return model, train_state
