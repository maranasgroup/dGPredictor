from scipy.io import savemat, loadmat
import pandas as pd
import pdb
import json
import numpy as np
from numpy import median, mean
from sklearn.linear_model import BayesianRidge, LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, LeaveOneOut
import pickle
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()

# from component_contribution.linalg import LINALG
# from component_contribution.compound_cacher import CompoundCacher


def linear_regression():
    # ac = loadmat('./data/dGPredictor_stereo.mat')
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
    # y = b.flatten()
    y = b

    reg = LinearRegression(fit_intercept=False).fit(X, y)
    # filename = './model/linearReg_ac_all_model.sav'
    # pickle.dump(reg, open(filename, 'wb'))
    # filename = './model/linearReg_ac_all_model.sav'
    # outfilename = '../cache/db_ac_all/result_linearReg.csv'
    # predict(filename,outfilename)
    # pdb.set_trace()
    predicted = reg.predict(X)

    print(reg.coef_)
    # plt.hist(reg.coef_[0][0:264], bins=50)
    plt.hist(reg.coef_[0][0:163], bins=50)
    # plt.xscale('log')
    plt.xlabel('$\Delta_g G^o$')
    plt.ylabel('Count')
    plt.savefig('./figures/linear_cc_groups.png')
    
    mse = mean_squared_error(y, predicted)
    r2 = r2_score(y, predicted)

    print('Mean squared error: %.2f'
        % mse)
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.4f'
        % r2)

    fig, ax = plt.subplots()
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1)
    ax.set_xlabel('Measured $\Delta_r G^o$')
    ax.set_ylabel('Predicted $\Delta_r G^o$')
    plt.figtext(.7, .2, "MSE = %.2f" % mse)
    plt.figtext(.7, .25, "$R^2$ = %.4f" % r2)
    plt.savefig('./figures/linear_regression_cc.png')

def ridge_regression():
    ac = loadmat('./data/dGPredictor_stereo.mat')

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
    # y = b.flatten()
    y = b

    # reg = LinearRegression(fit_intercept=False).fit(X, y)
    alphas = np.logspace(-6, 6, 200)
    reg = RidgeCV(alphas=alphas).fit(X, y)
    # filename = './model/linearReg_ac_all_model.sav'
    # pickle.dump(reg, open(filename, 'wb'))
    # filename = './model/linearReg_ac_all_model.sav'
    # outfilename = '../cache/db_ac_all/result_linearReg.csv'
    # predict(filename,outfilename)
    # pdb.set_trace()
    print(reg.alpha_)
    # print(reg.coef_)
    plt.hist(reg.coef_[0][0:264], bins=50, color = 'burlywood')
    # plt.xscale('log')
    plt.xlabel('$\Delta_g G^o$')
    plt.ylabel('Count')
    plt.savefig('./figures/ridge_groups.png')

    predicted = reg.predict(X)

    mse = mean_squared_error(y, predicted)
    r2 = r2_score(y, predicted)

    print('Mean squared error: %.2f'
        % mse)
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.4f'
        % r2)

    fig, ax = plt.subplots()
    ax.scatter(y, predicted, color = 'burlywood')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1,)
    ax.set_xlabel('Measured $\Delta_r G^o$')
    ax.set_ylabel('Predicted $\Delta_r G^o$')
    plt.figtext(.7, .2, "MSE = %.2f" % mse)
    plt.figtext(.7, .25, "$R^2$ = %.4f" % r2)
    plt.savefig('./figures/ridge_regression.png')
    # plt.show()

def linear_regression_cc():
    ac = loadmat('../cache/component_contribution_ac_all.mat')

    S = ac['train_S']

    G = ac['G']
    b = ac['b']
    # w = ac['w']
    
    # pdb.set_trace()

    m, n = S.shape
    assert G.shape[0] == m
    assert b.shape == (n, 1)

    # Apply weighing
    # W = np.diag(w.flat)
    # Linear regression for the reactant layer (aka RC)
    # inv_S, r_rc, P_R_rc, P_N_rc = LINALG._invert_project(S * W)

    P_R_rc = ac['P_R_rc']
    P_N_rc = ac['P_N_rc']

    XR = np.dot(P_R_rc,S)
    XN = np.dot(P_N_rc,S)

    XNTG = np.dot(XN.T,G)

    # X = STG
    X = np.concatenate((XR.T,XNTG),1)
    y = b.flatten()

    reg = LinearRegression(fit_intercept=False).fit(X, y)

    filename = './model/linearReg_ac_all_cc_model.sav'
    pickle.dump(reg, open(filename, 'wb'))

    outfilename = '../cache/db_ac_all/result_linearReg_cc.csv'
    predict_cc(filename,outfilename)


def test_decompse_rxn():
    molecular_signature = json.load(open('../cache/db_ac_all/decompose_vector_ac.json'))
    molsigs = pd.DataFrame.from_dict(molecular_signature).fillna(0)
    reactions_dict = json.load(open('../examples/optstoic_v3_Sji_dict.json'))
    reaction = reactions_dict['R00713']
    x, g = decompose_reaction(reaction,molsigs)

    # zeros = np.zeros((1, 44))

    # g = np.concatenate((g.T, zeros),1)

    ac = loadmat('../cache/component_contribution_ac_all.mat')

    dg = float(x.T*ac['dG0_cc'] + g.T*ac['dG0_gc'])
    print dg
    # X = np.concatenate((x.T, g.T),1)

    # filename = './model/linearReg_ac_all_cc_model.sav'
    # loaded_model = pickle.load(open(filename, 'rb'))

    # ymean = loaded_model.predict(X)
    # print ymean
    pdb.set_trace()


def decompose_reaction(reaction,molsigs):
    
    ac = loadmat('../cache/component_contribution_ac_all.mat')
    cids = list(ac['cids'])
    G = ac['G']

    # calculate the reaction stoichiometric vector and the group incidence
    # vector (x and g)
    Nc = len(cids)
    x = np.matrix(np.zeros((Nc, 1)))
    x_prime = []
    G_prime = []

    for compound_id, coeff in reaction.iteritems():
        if compound_id in cids:
            i = cids.index(compound_id)
            x[i, 0] = coeff
        else:
            # Decompose the compound and calculate the 'formation energy'
            # using the group contributions.
            # Note that the length of the group contribution vector we get
            # from CC is longer than the number of groups in "groups_data"
            # since we artifically added fictive groups to represent all the
            # non-decomposable compounds. Therefore, we truncate the
            # dG0_gc vector since here we only use GC for compounds which
            # are not in cids_joined anyway.
            x_prime.append(coeff)
            vector = molsigs['compound_id'].tolist()
            group_vec = np.array(vector)
            G_prime.append(group_vec)

    if x_prime != []:
        g = np.matrix(x_prime) * np.vstack(G_prime)
    else:
        g = np.matrix(np.zeros((1, 1)))

    g.resize((G.shape[1], 1))

    return x, g

def compare_coeff():
    ac = loadmat('../cache/component_contribution_ac_all.mat')
    dG_gc = ac['dG0_gc']

    filename = './model/linearReg_ac_all_cc_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    

    result = {}
    result['MATLAB'] = dG_gc.flatten()
    result['sklearn'] = loaded_model.coef_
    pdb.set_trace()
    df_result = pd.DataFrame.from_dict(result)
    df_result.to_csv('../cache/db_ac_all/compare_coeff_cc.csv')

def ridge_all_data():
    ac = loadmat('../cache/component_contribution_ac_all.mat')

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
    print reg.coef_
    # conv = reg.sigma_
    # conv_coeff = [conv[i][i] for i in range(len(conv))]

    # for num in conv_coeff[0:263]:
    #     if num < 500: print num
    # pdb.set_trace()
    filename = './model/bayesianRL_ac_all_model.sav'
    pickle.dump(reg, open(filename, 'wb'))

def get_rxn_rule():
    """calculate reaction rules based on the relaxed molecular signatures.

    Parameters
    ----------
    radius : int
        the radius is bond-distance that defines how many neighbor atoms should
        be considered in a reaction center.

    Returns
    -------
    None
        All of the reaction rules are saved in files (csv file)

    """
    reaction_dict = json.load(open('../examples/optstoic_v3_Sji_dict.json'))
    molecular_signature = json.load(open('../cache/db_ac_all/decompose_vector_ac.json'))
    molsigna_df = pd.DataFrame.from_dict(molecular_signature).fillna(0)
    all_mets = molsigna_df.columns.tolist()
    all_mets.append("C00080")
    all_mets.append("C00282")


    rule_df = pd.DataFrame(index=molsigna_df.index)
    for rid, value in reaction_dict.items():
        # skip the reactions with missing metabolites
        mets = value.keys()
        flag = False
        for met in mets:
            if met not in all_mets: 
                flag = True
                break
        if flag: continue

        rule_df[rid] = 0
        for met, stoic in value.items():
            if met == "C00080" or met == "C00282":
                continue  # hydogen is zero
            rule_df[rid] += molsigna_df[met] * stoic
    rule_df.to_csv("../cache/db_ac_all/relaxed_rule.csv", index=True)


def remove_duplicate():
    """Remove duplicated reaction rules from the all the rules generated from
    reactions directly.

    Returns
    -------
    None
        A new csv file is created to store the informaiton of unique reaction
        rules.

    """
    df = (
        pd.read_csv("../cache/db_ac_all/relaxed_rule.csv", index_col=0)
        .T.drop_duplicates()
        .T
    )
    df.to_csv("../cache/db_ac_all/relaxed_rule_noduplic.csv", index=True)


def remove_reversedRule():
    """in addition to remove the same rules, the reversed rules should also be
    removed. Note that this function is not well implemented because the loop is
    time consuming.

    Returns
    -------
    None
        A new csv file is created to store the informaiton of unique reaction
        rules.

    """
    # #
    df = (
        pd.read_csv("../cache/db_ac_all/relaxed_rule_noduplic.csv", index_col=0)
    )

    unique = []
    duplicate = []

    for col in df.columns.values:
        same_rules = getIdenticalRule(df, col)

        if len(same_rules) == 1:
            unique.append(col)
        else:
            same_rules.sort()
            if same_rules in duplicate:
                continue
            else:
                duplicate.append(same_rules)
    for sublist in duplicate:
        unique.append(sublist[0])

    new_df = df[unique]
    new_df.to_csv("./data/relaxed_rule_noduplic_v2.csv", index=True)

def predict(filename,outfilename):
    # filename = './model/bayesianRL_ac_all_model.sav'
    # outfilename = '../cache/db_ac_all/result.csv'
    loaded_model = pickle.load(open(filename, 'rb'))

    df = (
        pd.read_csv("../cache/db_ac_all/relaxed_rule_noduplic.csv", index_col=0)
    )

    # rule_list = df['R04734'].tolist()
    rule_vec = df.to_numpy().T

    m, n = rule_vec.shape
    
    zeros = np.zeros((m, 44))

    # rule_vec = np.asarray(rule_list)
    # pdb.set_trace()
    # X = np.concatenate([rule_vec,zeros])
    X = np.concatenate((rule_vec,zeros),1)

    # X = X.reshape(1,-1)
    # pdb.set_trace()
    # ymean, ystd = loaded_model.predict(X, return_std=True)

    ymean = loaded_model.predict(X)
    rxns = df.columns.tolist()
    # print(ymean)
    # print(ystd)
    result = {}
    result['reaction'] = rxns
    result['dG'] = ymean
    # result['dG_std'] = ystd
    df_result = pd.DataFrame.from_dict(result)
    df_result.to_csv(outfilename)

def predict_cc(filename,outfilename):
    # filename = './model/bayesianRL_ac_all_model.sav'
    # outfilename = '../cache/db_ac_all/result.csv'
    loaded_model = pickle.load(open(filename, 'rb'))

    df = (
        pd.read_csv("../cache/db_ac_all/relaxed_rule_noduplic.csv", index_col=0)
    )

    # rule_list = df['R04734'].tolist()
    rule_vec = df.to_numpy().T

    m, n = rule_vec.shape
    
    zeros = np.zeros((m, 44))

    # rule_vec = np.asarray(rule_list)
    # pdb.set_trace()
    # X = np.concatenate([rule_vec,zeros])
    X = np.concatenate((rule_vec,zeros),1)

    # X = X.reshape(1,-1)
    # pdb.set_trace()
    # ymean, ystd = loaded_model.predict(X, return_std=True)

    ymean = loaded_model.predict(X)
    rxns = df.columns.tolist()
    # print(ymean)
    # print(ystd)
    result = {}
    result['reaction'] = rxns
    result['dG'] = ymean
    # result['dG_std'] = ystd
    df_result = pd.DataFrame.from_dict(result)
    df_result.to_csv(outfilename)

def change_direction():
    df = (
        pd.read_csv("../cache/db_ac_all/relaxed_rule_noduplic.csv", index_col=0)
    )

    df_dG = pd.read_csv('../cache/db_ac_all/result.csv', index_col='reaction')
    dG = df_dG['dG'].to_dict()
    
    for rxn,value in dG.iteritems():
        if value > 0:
            df[rxn] = -1*df[rxn]

    df_new = df.T.drop_duplicates().T
    df_new.to_csv("../cache/db_ac_all/relaxed_rule_noduplic_v2.csv", index=True)
    # df.to_csv("../cache/db_ac_all/relaxed_rule_noduplic_negative.csv", index=True)

def predict_v2():
    filename = './model/bayesianRL_ac_all_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    df = (
        pd.read_csv("../cache/db_ac_all/relaxed_rule_noduplic_v2.csv", index_col=0)
    )

    # rule_list = df['R04734'].tolist()
    rule_vec = df.to_numpy().T

    m, n = rule_vec.shape
    
    zeros = np.zeros((m, 44))

    # rule_vec = np.asarray(rule_list)
    # pdb.set_trace()
    # X = np.concatenate([rule_vec,zeros])
    X = np.concatenate((rule_vec,zeros),1)

    # X = X.reshape(1,-1)
    # pdb.set_trace()
    ymean, ystd = loaded_model.predict(X, return_std=True)

    rxns = df.columns.tolist()
    # print(ymean)
    # print(ystd)
    result = {}
    result['reaction'] = rxns
    result['dG'] = ymean
    result['dG_std'] = ystd
    df_result = pd.DataFrame.from_dict(result)
    df_result.to_csv('../cache/db_ac_all/result_v2.csv')

def get_dG0_prime():    
    ccache = CompoundCacher()

    df = pd.read_csv('../cache/db_ac_all/result.csv',index_col='reaction')
    reactions_dict = json.load(open('../examples/optstoic_v3_Sji_dict.json'))

    pH = 7
    I = 0.1
    T = 298.15

    ddG0s = []
    for rxn in df.index.tolist():
        rxn_dict = reactions_dict[rxn]
        ddG0 = get_transform_ddG0(rxn_dict, ccache, pH, I, T)
        ddG0s.append(ddG0)
    df['ddG0'] = ddG0s
    df.to_csv('../cache/db_ac_all/result_v3.csv')

def get_transform_ddG0(rxn_dict, ccache, pH, I, T):
    """
    needed in order to calculate the transformed Gibbs energies of
    reactions.

    Returns:
        The difference between DrG0_prime and DrG0 for this reaction.
        Therefore, this value must be added to the chemical Gibbs
        energy of reaction (DrG0) to get the transformed value.
    """
    
    ddG0_forward = 0
    for compound_id, coeff in rxn_dict.iteritems():
        comp = ccache.get_compound(compound_id)
        ddG0_forward += coeff * comp.transform_pH7(pH, I, T)
    return ddG0_forward

def find_identical_rule():
    # rule = 'R00713' # CAR enzyme
    # rule = 'R01857' # zero changes
    # rule = 'R09281' # alcohol hodrogynase
    # rule = 'R01163'
    # rule = 'R03012'
    # rule = 'R05336'
    rule = 'R05804'
    df = pd.read_csv("../cache/db_ac_all/relaxed_rule.csv", index_col=0)
    identical_reactions = {}
    identical_reactions['forward'] = []
    identical_reactions['reverse'] = []
    for col in df.columns.values:
        if df[col].equals(df[rule]):
            # print col
            identical_reactions['forward'].append(col)
    for col in df.columns.values:
        if df[col].equals(-df[rule]):
            # print col
            identical_reactions['reverse'].append(col)
    print identical_reactions
    print len(identical_reactions['forward']) + len(identical_reactions['reverse'])  

if __name__ == '__main__':
    # linear_regression_cc()
    # test_decompse_rxn()
    # ridge_all_data()
    # get_rxn_rule()
    # remove_duplicate()
    # remove_reversedRule()
    # predict()
    # change_direction()
    # predict_v2()
    # find_identical_rule()
    # compare_coeff()
    # get_dG0_prime()
    linear_regression()
    # ridge_regression()