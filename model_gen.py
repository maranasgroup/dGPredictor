from scipy.io import savemat, loadmat
import pandas as pd
import pdb
import json
import numpy as np
from numpy import median, mean
from sklearn.linear_model import BayesianRidge, LinearRegression, RidgeCV, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, LeaveOneOut
import joblib
import pickle
import matplotlib.pyplot as plt
import sys
import os.path
import glob, os
import openbabel
from IPython.display import clear_output
import timeit 


ac = loadmat('./data/Test_KEGG_all_grp.mat')

y = ac['y']
y = y.flatten()

alphas = np.logspace(-6, 6, 200)

Xrc = ac['X_comb_all']
regr_rcombined = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True).fit(Xrc, y)

y_pred_rc = regr_rcombined.predict(Xrc)
mse_rc = mean_squared_error(y, y_pred_rc)
r2 = r2_score(y, y_pred_rc)


print('radius 1+2 linear model')
print('Mean squared error: %.2f'
    % mse_rc)
print('Coefficient of determination: %.4f'
    % r2)



s0 = timeit.default_timer()
joblib.dump(regr_rcombined,  './model/M12_model_BR.pkl',compress=3)
s1 = timeit.default_timer()
print(s1 - s0)

s0 = timeit.default_timer()
filename = './model/M12_model_BR.pkl'
loaded_model = joblib.load(open(filename, 'rb'))
s1 = timeit.default_timer()
print(s1 - s0)
print('==================================')
