import numpy as np
import deepxde as dde
import tensorflow as tf


def create_pinn_model(t_grid, solution_values, oxides):

    # initialize external trainable variables
    # v_initial = [dde.Variable(0.33, dtype="float64") for i in range(len(oxides))]

    v_01 = dde.Variable(0.33, dtype="float64")
    v_02 = dde.Variable(0.33, dtype="float64")
    v_03 = dde.Variable(0.33, dtype="float64")
    t_shift = dde.Variable(15.8, dtype="float64")
    baseline_h = dde.Variable(5, dtype="float64")
    t_beg = dde.Variable(1.6, dtype="float64")
    t_end = dde.Variable(2.2, dtype="float64")

    # helper functions
    def temperature(t):
        """Temperature function"""
        return t + (100 * t_shift)

    def temperature_derivative(_):
        """Temperature derivative"""
        return 1

    def f(t, k, e):
        """f(t) = K - E/T(t)"""
        return (k * 1e+1) - 1e+5 * e / temperature(t)

    def df_t(t, e):
        """f'(t) = (E * T'(t)) / (T(t))^2"""
        return (1e+5 * e * temperature_derivative(t)) / (temperature(t) ** 2)

    def baseline_function(t):
        """Baseline function"""
        return baseline_h / (
                1 + tf.exp(-4 * (temperature(t) - (t_beg * 1000 + t_end * 1000) / 2) / (t_end * 1000 - t_beg * 1000)))

    def ode(t, v):
        """ode system: v'(t) = v(t)(df_t - exp(f(t)))"""
        v1, v2, v3, baseline, v_sum = v[:, 0:1], v[:, 1:2], v[:, 2:3], v[:, 3:4], v[:, 4:]
        dv1_t = dde.grad.jacobian(v1, t)
        dv2_t = dde.grad.jacobian(v2, t)
        dv3_t = dde.grad.jacobian(v3, t)
        return [dv1_t - v1 * (df_t(t, oxides[0]["E"]) - tf.math.exp(f(t, oxides[0]["K"], oxides[0]["E"]))),
                dv2_t - v2 * (df_t(t, oxides[1]["E"]) - tf.math.exp(f(t, oxides[1]["K"], oxides[1]["E"]))),
                dv3_t - v3 * (df_t(t, oxides[2]["E"]) - tf.math.exp(f(t, oxides[2]["K"], oxides[2]["E"]))),
                baseline - baseline_function(t),
                v_sum - (v1 + v2 + v3 + baseline)]

    def transform_output(_, v):
        v1 = tf.math.exp(v[:, 0:1])
        v2 = tf.math.exp(v[:, 1:2])
        v3 = tf.math.exp(v[:, 2:3])
        baseline = v[:, 3:4]
        new_v = tf.reshape(tf.stack([v1, v2, v3, baseline, v1 + v2 + v3 + baseline], axis=1), (-1, 5))
        return new_v

    geom = dde.geometry.TimeDomain(0, 600)

    ic1 = dde.icbc.IC(geom, func=lambda t: v_01,
                      on_initial=lambda t, on_initial: np.isclose(t[0], oxides[0]["t_initial"]),
                      component=0)

    ic2 = dde.icbc.IC(geom, func=lambda t: v_02,
                      on_initial=lambda t, on_initial: np.isclose(t[0], oxides[1]["t_initial"]),
                      component=1)

    ic3 = dde.icbc.IC(geom, func=lambda t: v_03,
                      on_initial=lambda t, on_initial: np.isclose(t[0], oxides[2]["t_initial"]),
                      component=2)

    observe_v = dde.icbc.PointSetBC(t_grid, solution_values, component=4)
    t_grid_with_initial_t = np.concatenate([t_grid,
                                            np.array([oxides[0]["t_initial"],
                                                      oxides[1]["t_initial"],
                                                      oxides[2]["t_initial"]]).reshape(-1, 1)])
    data = dde.data.PDE(geom, ode,
                        [ic1, ic2, ic3, observe_v],
                        0, 0, anchors=t_grid_with_initial_t,
                        train_distribution="uniform")

    net = dde.nn.FNN([1] + [50] * 3 + [4], ["tanh"] * 3 + ["relu"], "Glorot uniform")
    net.apply_output_transform(transform_output)
    model = dde.Model(data, net)

    external_trainable_variables = [v_01, v_02, v_03, t_shift, baseline_h, t_beg, t_end]
    variable = dde.callbacks.VariableValue(external_trainable_variables, period=500, filename="variables.dat")

    return model, external_trainable_variables, variable
