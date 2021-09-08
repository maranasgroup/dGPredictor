import pandas as pd
import pulp
import pdb
import os
import json
from rdkit import Chem

# pulp_solver = pulp.solvers.CPLEX_CMD(path=None, keepFiles=0, mip=1, msg=1,
#      options=['mip tolerances mipgap 0', 'mip tolerances absmipgap 0',
#       'mip tolerances integrality 0', 'simplex tolerances optimality 1E-9',
#       'simplex tolerances feasibility 1E-9',], timelimit=1200)

def count_substructures(radius,molecule):
    """Helper function for get the information of molecular signature of a
    metabolite. The relaxed signature requires the number of each substructure
    to construct a matrix for each molecule.
    Parameters
    ----------
    radius : int
        the radius is bond-distance that defines how many neighbor atoms should
        be considered in a reaction center.
    molecule : Molecule
        a molecule object create by RDkit (e.g. Chem.MolFromInchi(inchi_code)
        or Chem.MolToSmiles(smiles_code))
    Returns
    -------
    dict
        dictionary of molecular signature for a molecule,
        {smiles: molecular_signature}
    """
    m = molecule
    smi_count = dict()
    atomList = [atom for atom in m.GetAtoms()]

    for i in range(len(atomList)):
        env = Chem.FindAtomEnvironmentOfRadiusN(m,radius,i)
        atoms=set()
        for bidx in env:
            atoms.add(m.GetBondWithIdx(bidx).GetBeginAtomIdx())
            atoms.add(m.GetBondWithIdx(bidx).GetEndAtomIdx())

        # only one atom is in this environment, such as O in H2O
        if len(atoms) == 0:
            atoms = {i}

        smi = Chem.MolFragmentToSmiles(m,atomsToUse=list(atoms),
                                    bondsToUse=env,canonical=True)

        if smi in smi_count:
            smi_count[smi] = smi_count[smi] + 1
        else:
            smi_count[smi] = 1
    return smi_count

def novoStoic_minFlux_relaxedRule(exchange_mets, novel_mets,project,iterations,pulp_solver,use_direction):
    """apply reaction rules generated from a more relaxed manner to search for
    reaction rules that are able to fill the gap between the source and sink
    metabolites.
    - rePrime procedure is more similar to a morgan fingerprints
    - the relaxed rule is generated from substructures without considering the
      bond that connect the atoms at the edge of the substructure to the rest
      of the molecules

    Parameters
    ----------
    exchange_mets : dict
        overall stoichiometry of source and sink metabolites, {met: stoic,...}
        This is a important input for novoStoic to run correctly because the
        method requires that overall moieties are balanced.
    novel_mets : list
        list of novel metabolites that are not in the database (novoStoic/data/
        metanetx_universal_model_kegg_metacyc_rhea_seed_reactome.json)
    filtered_rules : list
        list of rules that are filtered by the user (based on expert knowldedge)
        to reduce the running time of the novoStoic search process
    project : string
        a path to store the tmp information of result from running novoStoic
    iterations : int
        the number of iterations of searching for alternative solutions
    data_dir : type
        Description of parameter `data_dir`.

    Returns
    -------
    None
        all the outputs are saved in the project folder.

    """
    if not os.path.exists(project):
        os.makedirs(project)

    # the maximum flux of a reaction
    M = 2

    data_dir = './data'

    # read csv files with molecular signatures and reaction rules
    molecular_signature = json.load(open(
        os.path.join(data_dir, 'decompose_vector_ac.json')))
    molsigs = pd.DataFrame.from_dict(molecular_signature).fillna(0)

    rules = pd.read_csv(
        os.path.join(data_dir, "relaxed_rule_noduplic.csv"), index_col=0
    )

    ###### sets ############
    moiety_index = rules.index.tolist()  # moiety sets
    rules_index = rules.columns.values.tolist()
    print("Number of rules used in this search:",len(rules_index))

    exchange_index = exchange_mets.keys()

    ###### parameters ######
    # T(m,r) contains atom stoichiometry for each rule
    T = rules.to_dict(orient="index")

    # C(m,i) contains moiety cardinality for each metabolite
    C = molsigs.to_dict(orient="index")
    for m in moiety_index:
        C[m]["C00080"] = 0
        C[m]["C00282"] = 0

    # add metabolites that are not present in current database
    for met in novel_mets:
        # molsigs_product = pd.read_csv(
        #     project + "/relaxed_molsig_" + met + "_1.csv", index_col=0
        # )
        # molsigs_product_dict = molsigs_product.to_dict(orient="index")
        smiles = novel_mets[met]
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.RemoveHs(mol)
        molsigs_product_dict = count_substructures(1,mol)

        for m in moiety_index:
            if m in molsigs_product_dict.keys():
                C[m][met] = molsigs_product_dict[m]
            else:
                C[m][met] = 0

    ###### variables ######
    v_rule = pulp.LpVariable.dicts(
        "v_rule", rules_index, lowBound=-M, upBound=M, cat="Integer"
    )
    v_rule_obj = pulp.LpVariable.dicts(
        "v_rule_obj", rules_index, lowBound=0, upBound=M, cat="Continuous"
    )

    v_EX = pulp.LpVariable.dicts(
        "v_EX", exchange_index, lowBound=-M, upBound=M, cat="Continuous"
    )
    y_rule = pulp.LpVariable.dicts(
        "y", rules_index, lowBound=0, upBound=1, cat="Binary"
    )

    # create MILP problem
    lp_prob = pulp.LpProblem("novoStoic", pulp.LpMinimize)

    ####### objective function ####
    lp_prob += pulp.lpSum([v_rule_obj[j] for j in rules_index])

    ####### constraints ####
    # constraint 1: moiety change balance
    for m in moiety_index:
        lp_prob += (
            pulp.lpSum([T[m][r] * v_rule[r] for r in rules_index if T[m][r] !=0])
            == pulp.lpSum([C[m][i] * v_EX[i] for i in exchange_index if C[m][i] != 0]),
            "moiety_balance_" + str(moiety_index.index(m)),
        )

    # constraint 2: constraint for exchange reactions
    for i, stoic in exchange_mets.items():
        lp_prob += v_EX[i] == stoic, "exchange" + i

    # constraint 3: control the number of rules

    direction_df = pd.read_csv(
        os.path.join(data_dir, "direction.csv"), index_col=0
    )
    direction_df.index = direction_df['reaction']

    # direction: 0-reversible, 1-backward, 2-forward
    direction = direction_df['direction'].to_dict()

    if use_direction:
        soln_file = os.path.join(project, "solution_use_direction.txt")
        for j in rules_index:
            if direction[j] == 0:
                lp_prob += v_rule[j] >= y_rule[j] * -M, "cons1_%s" % j
                lp_prob += v_rule[j] <= y_rule[j] * M, "cons2_%s" % j
            if direction[j] == 1:
                lp_prob += v_rule[j] >= y_rule[j] * -M, "cons1_%s" % j
                lp_prob += v_rule[j] <= 0, "cons2_%s" % j
            if direction[j] == 2:
                lp_prob += v_rule[j] >= 0, "cons1_%s" % j
                lp_prob += v_rule[j] <= y_rule[j] * M, "cons2_%s" % j
    else:
        soln_file = os.path.join(project, "solution_no_direction.txt")
        for j in rules_index:
            lp_prob += v_rule[j] >= y_rule[j] * -M, "cons1_%s" % j
            lp_prob += v_rule[j] <= y_rule[j] * M, "cons2_%s" % j
    
    for j in rules_index:
        lp_prob += v_rule_obj[j] >= v_rule[j]
        lp_prob += v_rule_obj[j] >= -v_rule[j]
    
    # constraint 5: customized constraints
    # the number of steps of the pathway
    lp_prob += pulp.lpSum([v_rule_obj[j] for j in rules_index]) == 2

    ### solve
    integer_cuts(lp_prob,pulp_solver,iterations,rules_index,y_rule,v_rule,soln_file,direction)

def integer_cuts(lp_prob,pulp_solver,iterations,rules_index,y_rule,v_rule,soln_file,direction):
    """add integer cut constraints to a mixed-integer linear programming problem
    (MILP). The aim of such constraints is to find alternative solutions by
    adding constraints to exclude the already explored solutions.

    Reference: Optimization Methods in Metabolic Networks By Costas D. Maranas,
    Ali R. Zomorrodi, Chapter 4.2.2 Finding alternative optimal integer
    solutions

    Returns
    -------
    type
        Description of returned object.

    """
    for sol_num in range(1, iterations + 1):
        integer_cut_rules = []

        # optinal output: lp file for debug
        lp_prob.writeLP('./test.lp')
        # if pulp_solver = "SCIP":
        # status, values = pulp_solver.solve(lp_prob)
        lp_prob.solve(pulp_solver)
        # pulp_solver.solve(lp_prob)

        print("Status:", pulp.LpStatus[lp_prob.status])

        if pulp.LpStatus[lp_prob.status] != 'Optimal':
            break

        print('-----------rules--------------')
        with open(soln_file,'a') as f:
            f.write('iteration,' + str(sol_num))
            f.write('\n')

        for r in rules_index:
            if (v_rule[r].varValue >= 0.1 or v_rule[r].varValue <=-0.1):

                dG_info = ''
                if (v_rule[r].varValue > 0 and direction[r] == 1) or (v_rule[r].varValue < 0 and direction[r] == 2):
                    # print("##### Found ####: " + str(r))
                    # with open(soln_file,'a') as f:
                    #     f.write('##### Found ####: ' + str(r))
                    #     f.write('\n')
                    dG_info = ' * Thermodynamically infeasible'
                    print("##### Found ####: " + str(r) + dG_info)
                integer_cut_rules.append(r)
                print(r,v_rule[r].varValue)

                with open(soln_file,'a') as f:
                    f.write(r + ',' + str(v_rule[r].varValue) + dG_info)
                    f.write('\n')

        length = len(integer_cut_rules) - 1
        lp_prob += (
            pulp.lpSum([y_rule[r] for r in integer_cut_rules]) <= length,
            "integer_cut_" + str(sol_num),
        )


def test_bdo():
    exchange_mets = {
    'C00091': -1, # Succinyl-CoA
    'C00004': -4, # NADH
    'C00003': 4, # NAD+
    'C00010': 1, # coa
    'C00001':1, # h2O
    '14bdo': 1,
    }
    novel_mets = {
    '14bdo': 'OCCCCO'
    }

    iterations = 50
    project = './novoStoic_result'

#     path_to_cplex = '/Users/linuswang/Applications/IBM/ILOG/CPLEX_Studio1261/cplex/bin/x86-64_osx/cplex'
#     pulp_solver = pulp.CPLEX_CMD(path=path_to_cplex,keepFiles=0, mip=1, msg=1)

    pulp_solver = pulp.CPLEX_CMD(path=None,keepFiles=0, mip=1, msg=1)
    # pulp_solver = pulp.solvers.GUROBI_CMD()
    # pulp_solver = pulp.solvers.GLPK_CMD()
    use_direction=True
    novoStoic_minFlux_relaxedRule(exchange_mets, novel_mets,project,iterations,pulp_solver,use_direction)
    use_direction=False
    novoStoic_minFlux_relaxedRule(exchange_mets, novel_mets,project,iterations,pulp_solver,use_direction)


def test_isovalarate():
    exchange_mets = {
        'C00141': -1, # 2-keto isovalarate
        'C00004': -1, # NADH
        'C00003': 1, # NAD+
        "C14710": 1, # isobutanol C4H10O
        'C00011': 1, # co2
    }
    novel_mets = {}

    iterations = 50
    project = './novoStoic_isovalarate'

#     path_to_cplex = '/Users/linuswang/Applications/IBM/ILOG/CPLEX_Studio1261/cplex/bin/x86-64_osx/cplex'
#     pulp_solver = pulp.CPLEX_CMD(path=path_to_cplex,keepFiles=0, mip=1, msg=1)

    pulp_solver = pulp.CPLEX_CMD(path=None,keepFiles=0, mip=1, msg=1)
    # pulp_solver = pulp.solvers.GUROBI_CMD()
    # pulp_solver = pulp.GLPK_CMD()
    # use_direction=True
    # novoStoic_minFlux_relaxedRule(exchange_mets, novel_mets,project,iterations,pulp_solver,use_direction)
    use_direction=False
    novoStoic_minFlux_relaxedRule(exchange_mets, novel_mets,project,iterations,pulp_solver,use_direction)

if __name__ == '__main__':
    test_isovalarate()
