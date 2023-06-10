import numpy as np
from examples.functions import create_oxide_function, create_baseline_function
from examples.test_three_oxides_with_baseline import test_inverse_problem, create_reference_solution
from examples.oxide_examples import good as oxides_examples
from examples.generate_oxides import find_t_initial
from utils.visualiser import plot_peaks, plot_prediction


def generate_random_baseline():
    h = np.random.normal(loc=1.5, scale=0.5)
    t_beg = np.random.normal(loc=1700, scale=100)
    t_end = np.random.normal(loc=2100, scale=100)
    return {"h": h, "t_beg": t_beg, "t_end": t_end}


def generate_random_t_shift():
    res =  np.random.normal(loc=1580, scale=10)
    return res


def generate_random_solution(t_shift):
    number_of_oxides = np.random.randint(3, 5)
    # number_of_oxides = 4
    oxides_params = np.random.choice(oxides_examples, size=number_of_oxides, replace=False).tolist()
    oxides_concentrations = np.random.normal(loc=0.5, scale=0.1, size=number_of_oxides)
    oxides_concentrations[oxides_concentrations < 0] = 0

    for oxide_concentration, oxide_params in zip(oxides_concentrations, oxides_params):
        oxide_params["v_initial"] = oxide_concentration
        oxide_params["t_initial"] = find_t_initial(oxide_params["K"], oxide_params["E"], t_shift)

    return oxides_params


def test_model(sigma=0.0):
    t_shift = generate_random_t_shift()
    oxides = generate_random_solution(t_shift)
    baseline = generate_random_baseline()
    print(oxides, baseline, t_shift)
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
    _, solution_values = create_reference_solution(sigma, oxides, baseline, t_shift=t_shift,
                                                   number_of_points=number_of_points)
    plot_peaks(t_grid, [oxide(t_grid) for oxide in oxide_functions], baseline_func(t_grid), solution_values)
    true_variables = []
    for oxide in oxides:
        true_variables.append(oxide["v_initial"])
    true_variables += [t_shift]
    true_variables += baseline
    model, train_state, variables = test_inverse_problem(sigma, oxides, baseline, t_shift, use_lbfgs=False)

    true_variables = np.array(true_variables[:len(oxides)])
    predicted_variables = np.array(variables.value[:len(oxides)])
    print(np.sqrt(np.sum((true_variables - predicted_variables) ** 2)))
    # print(variable_deviation(true_variables, variables))

    plot_prediction(train_state.X_train, [train_state.y_pred_train[:, i] for i in range(len(oxides))], train_state.y_pred_train[:, len(oxides)], train_state.y_pred_train[:, len(oxides) + 1])
