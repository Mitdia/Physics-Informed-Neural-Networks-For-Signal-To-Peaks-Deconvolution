
import numpy as np
import deepxde as dde
import torch
import types

def forward(self, inputs):
    # Модификаци forward() в сети для добавления BatchNorm и активации в конце
#     print(inputs.shape)                             # +
    x = inputs
    if self._input_transform is not None:
        x = self._input_transform(x)
    for j, linear in enumerate(self.linears[:-1]):
        output = linear(x)  # +
        if hasattr(self, "bn"):
            output = self.bn[j](output)                 # +
        x = (
            self.activation[j](output)
            if isinstance(self.activation, list)
            else self.activation(output)
        )
    x = self.linears[-1](x)
    if hasattr(self, "last_activation"):            # +
        x = self.last_activation(x)                 # +
    if self._output_transform is not None:
        x = self._output_transform(inputs, x)
    return x


def hook(module, x):
    # Перевод типов входов в float32 или float64 в зависимости от типа сети (сама она это не делает на каких-то версиях pytorch, есть даже issue на GitHub'е)
    return tuple(t.to(dde.config.real(torch)) for t in x)


def create_pinn_model(t_grid, solution_values, oxides):
    # initialize external trainable variables
    num_oxides = len(oxides)
    v_initial = [dde.Variable(0.33, dtype=torch.double) for _ in range(num_oxides)]

    t_shift = dde.Variable(158.0, dtype=torch.double)

    baseline_h = dde.Variable(1.5, dtype=torch.double)
    t_beg = dde.Variable(1.6, dtype=torch.double)
    t_end = dde.Variable(2.2, dtype=torch.double)

    # helper functions
    def temperature(t):
        """Temperature function"""
        return t + (10 * t_shift)

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
                1 + torch.exp(-4 * (temperature(t) - (t_beg * 1000 + t_end * 1000) / 2) / (t_end * 1000 - t_beg * 1000)))

    def initial_condition_gen(oxide_index):
        def initial_function(_):
            return v_initial[oxide_index]

        def close_to_initial(t, _):
            return np.isclose(t[0], oxides[oxide_index]["t_initial"])

        return initial_function, close_to_initial

    def ode(t, v):
        """ode system: v'(t) = v(t)(df_t - exp(f(t)))"""
        oxide_functions = [v[:, j:j + 1] for j in range(num_oxides)]
        oxides_functions_derivatives = [dde.grad.jacobian(oxide_functions[j], t) for j in range(num_oxides)]
        baseline, v_sum = v[:, num_oxides:num_oxides + 1], v[:, num_oxides + 1:]
        oxides_odes = [oxides_functions_derivatives[j] - oxide_functions[j] * (
                df_t(t, oxides[j]["E"]) - torch.exp(f(t, oxides[j]["K"], oxides[j]["E"]))) for j in
                       range(num_oxides)]
        oxide_functions_sum = 0
        for j in range(num_oxides):
            oxide_functions_sum += oxide_functions[j]
        other_equations = [baseline - baseline_function(t),
                           v_sum - (oxide_functions_sum + baseline)]
        return oxides_odes + other_equations

    def transform_output(_, v):
        oxide_functions = [torch.exp(v[:, j:j + 1]) for j in range(num_oxides)]
        baseline = v[:, num_oxides:num_oxides + 1]
        oxide_functions_sum = 0
        for j in range(num_oxides):
            oxide_functions_sum += oxide_functions[j]
        new_v = torch.reshape(torch.stack(oxide_functions + [baseline, oxide_functions_sum + baseline], axis=1), (-1, num_oxides + 2))
        return new_v

    geom = dde.geometry.TimeDomain(0, 650)

    initial_conditions = []
    for i in range(num_oxides):
        initial_condition, is_close_to_initial = initial_condition_gen(i)
        initial_conditions.append(dde.icbc.IC(geom, func=initial_condition,
                                              on_initial=is_close_to_initial,
                                              component=i))

    observed_v = dde.icbc.PointSetBC(t_grid, solution_values, component=num_oxides + 1)
    t_grid_with_initial_t = np.concatenate([t_grid,
                                            np.array([oxides[i]["t_initial"] for i in range(num_oxides)]).reshape(-1,
                                                                                                                  1)])
    initial_conditions.append(observed_v)
    print(initial_conditions)
    data = dde.data.PDE(geom, ode,
                        initial_conditions,
                        0, 0, anchors=t_grid_with_initial_t,
                        train_distribution="uniform")

    net = dde.nn.FNN([1] + [50] * 3 + [num_oxides + 1], ["tanh"] * 4, "Glorot uniform")
    net.apply_output_transform(transform_output)
    net.bn = torch.nn.ModuleList([torch.nn.BatchNorm1d(x.out_features) for x in net.linears[:-1]])  # batchNorm
    # net.last_activation = torch.nn.ReLU()  # Sigmoid на выходе
    net.forward = types.MethodType(forward, net)  # перегрузка метода

    model = dde.Model(data, net)

    external_trainable_variables = v_initial + [t_shift, baseline_h, t_beg, t_end]
    variable = dde.callbacks.VariableValue(external_trainable_variables, period=500, filename="variables.dat")

    return model, external_trainable_variables, [variable]
