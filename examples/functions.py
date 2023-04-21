import numpy as np
from scipy.integrate import quad


def create_oxide_function(k, e, v_initial, t_initial, t_shift=1600):

    def f(t):
        """f(t) = K - E/T(t)"""
        return k - e / (t_shift + t)

    def function_for_integration(tau):
        return np.exp(f(tau))

    def oxide_function(t):
        """V(t) = V_0 e^(f(t) - f(t_0)) * e^(-integral(t_0, t)(e^f(tau)d tau))"""
        result = np.zeros(t.shape)
        for i in range(len(t)):
            integral = quad(function_for_integration, t_initial, t[i])[0]
            result[i] = np.exp(-integral)
        return v_initial * np.exp(f(t) - f(t_initial)) * result

    return oxide_function


def create_baseline_function(h, t_beg, t_end, t_shift=1600):

    def baseline_function(t):
        """f_b(T) = H / (1 + exp(-4(T - (T_end + T_beg) / 2)/ (T_end - T_beg)))"""
        return h / (1 + np.exp(-4 * ((t_shift + t) - (t_beg + t_end) / 2) / (t_end - t_beg)))

    return baseline_function


def create_solution(functions_list):

    def solution(t):
        result = 0
        for function in functions_list:
            result += function(t)
        return result

    return solution
