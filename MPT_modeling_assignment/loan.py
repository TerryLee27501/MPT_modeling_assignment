import pandas as pd
import numpy as np
from scipy.optimize import minimize


covariance_matrix = pd.read_csv('covariance_matrix.csv', header=None).values
expected_returns = np.array([0.006725673, 0.013520658, -0.027850174, 0.002436821, 0.042551583, 0.015047136, -0.033993402, -0.032995446, 0.08581829, 0.004635975, 0.0045, -0.008])
ini_constant_list = np.linspace(0.0045, 0.1045, 20)
constant_list = []
std_error_list = []


# 制定目标函数
def obj_func(weights):
    risk_weights = weights[:-2]
    return np.sqrt(np.dot(risk_weights.T, np.dot(covariance_matrix, risk_weights)))


for constant in ini_constant_list:
    # 初始化权重
    initial = [1/len(expected_returns) for _ in range(len(expected_returns))]
    # 设定约束条件
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 2 * x[-1] - 1},  # sum of weights must be 1
               {'type': 'eq', 'fun': lambda x: np.sum(x * (expected_returns)) - constant},  # expected returns must be equal to constant
               {'type': 'ineq', 'fun': lambda x: x[-2:]})  # weights must be non-negative
    result = minimize(obj_func, initial, method='SLSQP', constraints=constraints)
    if result.success:
        constant_list.append(constant)
        std_error_list.append(np.sqrt(np.dot(result.x[:-2].T, np.dot(covariance_matrix, result.x[:-2]))))


combined_array = np.column_stack((std_error_list, constant_list))
np.savetxt('loan.csv', combined_array, delimiter=',')