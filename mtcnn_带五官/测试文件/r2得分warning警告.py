'''
R2得分的定义：R^2 = 1 - (total_sum_squares)/(residual_sum_squares)
'''


import numpy as np
from sklearn.metrics import r2_score

x = np.array([2.3])
y = np.array([2.1]) # exact values do not matter

ret = r2_score(x, y)
print(ret)
