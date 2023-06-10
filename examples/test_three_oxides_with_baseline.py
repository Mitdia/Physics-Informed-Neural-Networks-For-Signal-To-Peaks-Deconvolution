from examples.functions import create_oxide_function, create_baseline_function, create_solution
from models.three_oxides_and_baseline import create_pinn_model
import numpy as np
import deepxde as dde


def create_reference_solution(sigma, solution_oxides, baseline_parameters, t_shift=1600, number_of_points=3000):
    oxide_functions = []

    t_grid = np.linspace(0, 650, number_of_points).reshape(-1, 1).astype("float64")
    for oxide in solution_oxides:
        oxide_functions.append(create_oxide_function(oxide["K"] * 1e1,
                                                     oxide["E"] * 1e5,
                                                     oxide["v_initial"],
                                                     oxide["t_initial"],
                                                     t_shift))

    baseline_func = create_baseline_function(**baseline_parameters, t_shift=t_shift)

    solution = create_solution(oxide_functions + [baseline_func])
    solution_values = solution(t_grid).astype("float64") + np.random.normal(0, sigma, number_of_points).reshape(-1, 1)

    return t_grid, solution_values



def test_inverse_problem(sigma, oxides, baseline_parameters, t_shift=1600, use_lbfgs=True):

    num_oxides = len(oxides)
    # create reference solution
    t_grid, solution_values = create_reference_solution(sigma, oxides, baseline_parameters, t_shift=t_shift)

    # create model
    model, external_trainable_variables, callbacks = create_pinn_model(t_grid, solution_values, oxides)

    # train adam
    loss_weights = [1e+5] * num_oxides + [0, 1e-0] + [1e-0] * num_oxides + [5e+2]
    model.compile("adam", lr=1e-0, loss_weights=loss_weights,
                  external_trainable_variables=external_trainable_variables)
    loss_history, train_state = model.train(iterations=100000, callbacks=callbacks)
    dde.saveplot(loss_history, train_state, issave=True, isplot=True)

    # train lbfgs
    if use_lbfgs:
        model.compile("L-BFGS", external_trainable_variables=external_trainable_variables)
        loss_history, train_state = model.train(callbacks=callbacks)
        dde.saveplot(loss_history, train_state, issave=True, isplot=True)

    return model, train_state, callbacks[0]
