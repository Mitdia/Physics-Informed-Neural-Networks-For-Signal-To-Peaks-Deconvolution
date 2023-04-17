from examples.test_three_oxides_with_baseline import test_inverse_problem
import deepxde as dde
import matplotlib as mpl

dde.config.set_random_seed(42)
dde.config.set_default_float("float64")


params = {"k_1": 4.0, "e_1": 0.9, "v_01_true": 0.6, "t_01": 300.0,
          "k_2": 4.0, "e_2": 0.75, "v_02_true": 0.2, "t_02": 0.0,
          "k_3": 3.3, "e_3": 0.7, "v_03_true": 0.3, "t_03": 150}

test_inverse_problem(0.1, params)
