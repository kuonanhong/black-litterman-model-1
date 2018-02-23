import numpy as np
import pandas as pd
from scipy.optimize import minimize

corr_matrix=pd.read_pickle('plk/corr_matrix').as_matrix()
re_arr=pd.read_pickle('plk/mean_series.plk').as_matrix()

# input: X is a n*1 numpy matrix, A is the N*N numpy corr
def func(X):
    x_tran=X.transpose()
    return np.dot(np.dot(x_tran,corr_matrix),X)
def func_deriv(X):
    x_tran=X.transpose()
    a_tran=corr_matrix.transpose()
    return np.dot(a_tran,x_tran)+np.dot(corr_matrix,X)
# the summation adds up to 1
def cons1(X):
    return np.sum(X)-1
# the return greater thatn 5%
def cons2(X):
    return np.dot(X.transpose(),re_arr)-0.05

W0=np.ones(re_arr.size)/re_arr.size
bnds=((0,0.4),)
temp=((0,0.4),)
for i in range(re_arr.size-1):
    bnds=bnds+temp
cons=({'type':'eq','fun':cons1},{'type':'ineq','fun':cons2})
result=minimize(func,x0=W0,jac=func_deriv,constraints=cons,bounds=bnds)
print(result)
