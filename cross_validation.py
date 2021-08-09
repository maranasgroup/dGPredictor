from scipy.io import savemat, loadmat
import pandas as pd
import pdb
import json
import numpy as np
from numpy import median, mean
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge, BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, LeaveOneOut
import pickle
import matplotlib.pyplot as plt

def getDuplicateColumns(otherCol,df):
    '''
    Get a list of duplicate columns.
    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.
    :param df: Dataframe object
    :return: List of columns whose contents are duplicates.
    '''
    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
    # Select column at xth index.
        col = df.iloc[:, x]
        if col.equals(otherCol):
            duplicateColumnNames.add(df.columns.values[x])
    return list(duplicateColumnNames)

def get_median():
    ac = loadmat('./data/component_contribution_python.mat')
    print(ac['train_S'].shape)

    df_S = pd.DataFrame(ac['train_S'])
    df_S_unique = df_S.T.drop_duplicates().T
    b = ac['b'].flat

    unque_cols = df_S_unique.columns.values.tolist()

    median_b = []

    for idx in unque_cols:
        # print idx
        dup_cols = getDuplicateColumns(df_S_unique[idx],df_S)

        values = []
        for col in dup_cols:
            value = b[col]
            values.append(value)
        
        median_b.append(median(values))
        # print median_b

    print(median_b)
    with open('./data/median_b.json','w') as fp:
        json.dump(median_b,fp)

def sklearn_regression():
    # ac = loadmat('../cache/component_contribution_python.mat')
    # ac = loadmat('./data/component_contribution_python.mat')

    S = ac['train_S']
    G = ac['G']
    b = ac['b']
    
    m, n = S.shape
    assert G.shape[0] == m
    assert b.shape == (n, 1)

    STG = np.dot(S.T,G)

    X = STG
    y = b
    reg = LinearRegression(fit_intercept=False).fit(X, y)

    y_pred = reg.predict(X)
    print(('Mean squared error: %.2f'
      % mean_squared_error(y, y_pred)))

def cross_validation_cc():
    ac = loadmat('./data/component_contribution_python.mat')

    S = ac['train_S']

    df_S = pd.DataFrame(ac['train_S'])
    df_S_unique = df_S.T.drop_duplicates().T
    unque_cols = df_S_unique.columns.values.tolist()
    S = S[:, unque_cols]

    G = ac['G']
    # b = ac['b']

    b_list = json.load(open('./data/median_b.json'))
    b = np.asarray(b_list)
    b = np.reshape(b,(-1,1))

    # w = ac['w']
    
    # pdb.set_trace()

    m, n = S.shape
    assert G.shape[0] == m
    assert b.shape == (n, 1)

    STG = np.dot(S.T,G)

    X = STG
    y = b
    
    # reg = LinearRegression(fit_intercept=False).fit(X, y)

    # y_pred = reg.predict(X)
    # print('Mean squared error: %.2f'
    #   % mean_squared_error(y, y_pred))
    # print('R2',reg.score(X, y))
    # # compare dG_gc with matlab
    # print reg.coef_

    # cross validation
    regression = LinearRegression(fit_intercept=False)
    # lasso = linear_model.Lasso()
    scores = -cross_val_score(regression, X, y, cv=LeaveOneOut(), scoring='neg_mean_absolute_error')
    # print scores
    # pdb.set_trace()
    print(('median of cv is: ', median(scores)))
    print(('mean of cv is: ', mean(scores)))

    print(('std of cv is: ', scores.std))
    x = np.sort(scores)
    # y = np.arange(1,len(x)+1)/len(x)
    y = 1. * np.arange(len(x)) / (len(x) - 1)

    fig = plt.figure(figsize=(6,6))
    plt.xlim(right=15)
    plt.plot(x,y,marker='.',linestyle='none')#,color="#273c75")
    plt.axhline(y=0.5,linewidth=1,color='grey')
    plt.xlabel('|$\Delta G^{\'o}_{est} - \Delta G^{\'o}_{obs}$|')
    plt.ylabel('Cumulative distribution')
    fig.savefig('./figures/cross_validation_cc.jpg')
    plt.show()

def cross_validation_ac_ridge():
    ac = loadmat('./data/dGPredictor_stereo.mat')

    S = ac['train_S']

    df_S = pd.DataFrame(ac['train_S'])
    df_S_unique = df_S.T.drop_duplicates().T
    unque_cols = df_S_unique.columns.values.tolist()
    S = S[:, unque_cols]

    G = ac['G']
    # b = ac['b']

    b_list = json.load(open('./data/median_b.json'))
    b = np.asarray(b_list)
    b = np.reshape(b,(-1,1))

    # w = ac['w']
    
    # pdb.set_trace()

    m, n = S.shape
    assert G.shape[0] == m
    assert b.shape == (n, 1)

    STG = np.dot(S.T,G)

    X = STG
    y = b

    alphas = np.logspace(-6, 6, 200)

    clf = RidgeCV(alphas=alphas, fit_intercept=False).fit(X, y)

    clf_new = Ridge(alpha=clf.alpha_,fit_intercept=False)
    # y_pred = clf.predict(X)
    scores = -cross_val_score(clf_new, X, y, cv=LeaveOneOut(), scoring='neg_mean_absolute_error')
    # scores = -cross_val_score(clf_new, X, y, cv=LeaveOneOut(), scoring='neg_mean_squared_error')
    # # print scores
    # # pdb.set_trace()
    print(('median of cv is: ', median(scores)))
    print(('mean of cv is: ', mean(scores)))

    x = np.sort(scores)
    # # y = np.arange(1,len(x)+1)/len(x)
    y = 1. * np.arange(len(x)) / (len(x) - 1)

    fig = plt.figure(figsize=(6,6))
    plt.xlim(right=15)
    plt.plot(x,y,marker='.',linestyle='none',color="burlywood")
    plt.axhline(y=0.5,linewidth=1,color='grey')
    plt.xlabel('|$\Delta G^{\'o}_{est} - \Delta G^{\'o}_{obs}$|')
    plt.ylabel('Cumulative distribution')
    fig.savefig('./figures/cross_validation_ridge.jpg')
    # plt.show()


def ridge():
    ac = loadmat('./data/component_contribution_python.mat')

    S = ac['train_S']

    df_S = pd.DataFrame(ac['train_S'])
    df_S_unique = df_S.T.drop_duplicates().T
    unque_cols = df_S_unique.columns.values.tolist()
    S = S[:, unque_cols]

    G = ac['G']
    # b = ac['b']

    b_list = json.load(open('./data/median_b.json'))
    b = np.asarray(b_list)
    b = np.reshape(b,(-1,1))

    # w = ac['w']
    
    # pdb.set_trace()

    m, n = S.shape
    assert G.shape[0] == m
    assert b.shape == (n, 1)

    STG = np.dot(S.T,G)

    X = STG
    y = b

    # clf = Ridge(alpha=0.1,fit_intercept=False)
    # clf.fit(X, y)
    # print('R2',clf.score(X, y))
    # print clf.coef_

    reg = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
    reg.fit(X,y)
    # print reg.coef_
    conv = reg.sigma_
    conv_coeff = [conv[i][i] for i in range(len(conv))]

    for num in conv_coeff[0:263]:
        if num < 500: print(num)
    pdb.set_trace()

def ridge_all_data():
    ac = loadmat('./data/component_contribution_python.mat')

    S = ac['train_S']

    G = ac['G']
    b = ac['b']

    # w = ac['w']
    
    # pdb.set_trace()

    m, n = S.shape
    assert G.shape[0] == m
    assert b.shape == (n, 1)

    STG = np.dot(S.T,G)

    X = STG
    y = b.flatten()

    # clf = Ridge(alpha=0.1,fit_intercept=False)
    # clf.fit(X, y)
    # print('R2',clf.score(X, y))
    # print clf.coef_

    reg = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
    reg.fit(X,y)
    print(reg.coef_)
    # conv = reg.sigma_
    # conv_coeff = [conv[i][i] for i in range(len(conv))]

    # for num in conv_coeff[0:263]:
    #     if num < 500: print num
    # pdb.set_trace()
    filename = './model/bayesianRL_model.sav'
    pickle.dump(reg, open(filename, 'wb'))


if __name__ == '__main__':
    # get_median()
    # cross_validation_ac()
    cross_validation_cc()
    cross_validation_ac_ridge()