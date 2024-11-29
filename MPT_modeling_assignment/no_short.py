import pandas as pd
import numpy as np
from scipy.optimize import minimize


covariance_matrix = pd.read_csv('covariance_matrix.csv', header=None).values
expected_returns = np.array([0.006725673, 0.013520658, -0.027850174, 0.002436821, 0.042551583, 0.015047136, -0.033993402, -0.032995446, 0.08581829, 0.004635975])
ini_constant_list = np.linspace(-0.04, 0.04, 20)
constant_list = []
std_error_list = []


def obj_func_sharpe(weights, constant):
    returns = np.sum(weights * expected_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    sharpe_ratio = (returns - constant) / volatility
    return sharpe_ratio


def obj_func(weights):
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))


for constant in ini_constant_list:
    initial = [1/len(expected_returns) for _ in range(len(expected_returns))]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # sum of weights must be 1
               {'type': 'eq', 'fun': lambda x: np.sum(x * expected_returns) - constant},  # expected returns must be equal to constant
               {'type': 'ineq', 'fun': lambda x: x})  # weights must be non-negative
    result = minimize(obj_func, initial, method='SLSQP', constraints=constraints)
    if result.success:
        constant_list.append(constant)
        std_error_list.append(np.sqrt(np.dot(result.x.T, np.dot(covariance_matrix, result.x))))


combined_array = np.column_stack((std_error_list, constant_list))
np.savetxt('no_short.csv', combined_array, delimiter=',')