from examples.test_three_oxides_with_baseline import test_inverse_problem
import deepxde as dde


dde.config.set_random_seed(42)
dde.config.set_default_float("float64")


oxides = [{"K": 4.0, "E": 0.9, "v_initial": 0.6, "t_initial": 300.0},
          {"K": 4.0, "E": 0.75, "v_initial": 0.2, "t_initial": 0.0},
          {"K": 3.3, "E": 0.7, "v_initial": 0.3, "t_initial": 150}]

baseline = {"h": 2, "t_beg": 1800, "t_end": 2000}

test_inverse_problem(0.1, oxides, baseline)
