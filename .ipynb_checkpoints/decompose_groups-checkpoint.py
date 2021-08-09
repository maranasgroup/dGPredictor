import pandas as pd
import pdb
import json
from rdkit import Chem

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

def decompse_ac(db_smiles,radius=1):
    non_decomposable = []
    decompose_vector = dict()

    for cid in db_smiles:
        # print cid
        smiles_pH7 = db_smiles[cid]
        try:
            mol = Chem.MolFromSmiles(smiles_pH7)
            mol = Chem.RemoveHs(mol)
            # Chem.RemoveStereochemistry(mol) 
            smi_count = count_substructures(radius,mol)
            decompose_vector[cid] = smi_count

        except Exception as e:
            non_decomposable.append(cid)

    with open('./data/decompose_vector_ac.json','w') as fp:
        json.dump(decompose_vector,fp)

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
    reaction_dict = json.load(open('./data/optstoic_v3_Sji_dict.json'))
    molecular_signature = json.load(open('./data/decompose_vector_ac.json'))
    molsigna_df = pd.DataFrame.from_dict(molecular_signature).fillna(0)
    all_mets = molsigna_df.columns.tolist()
    all_mets.append("C00080")
    all_mets.append("C00282")


    rule_df = pd.DataFrame(index=molsigna_df.index)
    for rid, value in list(reaction_dict.items()):
        # skip the reactions with missing metabolites
        mets = list(value.keys())
        flag = False
        for met in mets:
            if met not in all_mets: 
                flag = True
                break
        if flag: continue

        rule_df[rid] = 0
        for met, stoic in list(value.items()):
            if met == "C00080" or met == "C00282":
                continue  # hydogen is zero
            rule_df[rid] += molsigna_df[met] * stoic
    rule_df.to_csv("./data/reaction_rule.csv", index=True)

def get_rxn_rule_no_stero():
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
    reaction_dict = json.load(open('./data/optstoic_v3_Sji_dict.json'))
    molecular_signature = json.load(open('./data/decompose_vector_ac_nostereo.json'))
    molsigna_df = pd.DataFrame.from_dict(molecular_signature).fillna(0)
    all_mets = molsigna_df.columns.tolist()
    all_mets.append("C00080")
    all_mets.append("C00282")


    rule_df = pd.DataFrame(index=molsigna_df.index)
    for rid, value in list(reaction_dict.items()):
        # skip the reactions with missing metabolites
        mets = list(value.keys())
        flag = False
        for met in mets:
            if met not in all_mets: 
                flag = True
                break
        if flag: continue

        rule_df[rid] = 0
        for met, stoic in list(value.items()):
            if met == "C00080" or met == "C00282":
                continue  # hydogen is zero
            rule_df[rid] += molsigna_df[met] * stoic
    rule_df.to_csv("./data/reaction_rule_no_stero.csv", index=True)

def get_rxn_rule_remove_TECRDB_mets():
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
    reaction_dict = json.load(open('./data/optstoic_v3_Sji_dict.json'))
    molecular_signature = json.load(open('./data/decompose_vector_ac.json'))
    molsigna_df = pd.DataFrame.from_dict(molecular_signature).fillna(0)
    all_mets = molsigna_df.columns.tolist()
    all_mets.append("C00080")
    all_mets.append("C00282")

    mets_TECRDB_df = pd.read_csv('./data/TECRBD_mets.txt',header=None)
    mets_TECRDB = mets_TECRDB_df[0].tolist()

    # pdb.set_trace()
    all_mets = list(set(all_mets + mets_TECRDB))

    rule_df = pd.DataFrame(index=molsigna_df.index)
    for rid, value in list(reaction_dict.items()):
        # skip the reactions with missing metabolites
        mets = list(value.keys())
        flag = False
        for met in mets:
            if met not in all_mets: 
                flag = True
                break
        if flag: continue

        rule_df[rid] = 0
        for met, stoic in list(value.items()):
            if met in mets_TECRDB:
                continue  # hydogen is zero
            rule_df[rid] += molsigna_df[met] * stoic
    rule_df.to_csv("./data/reaction_rule_remove_TECRDB_mets.csv", index=True)

def get_rxn_rule_no_stero_remove_TECRDB_mets():
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
    reaction_dict = json.load(open('./data/optstoic_v3_Sji_dict.json'))
    molecular_signature = json.load(open('./data/decompose_vector_ac_nostereo.json'))
    molsigna_df = pd.DataFrame.from_dict(molecular_signature).fillna(0)
    all_mets = molsigna_df.columns.tolist()
    all_mets.append("C00080")
    all_mets.append("C00282")

    mets_TECRDB_df = pd.read_csv('./data/TECRBD_mets.txt',header=None)
    mets_TECRDB = mets_TECRDB_df[0].tolist()

    # pdb.set_trace()
    all_mets = list(set(all_mets + mets_TECRDB))

    rule_df = pd.DataFrame(index=molsigna_df.index)
    for rid, value in list(reaction_dict.items()):
        # skip the reactions with missing metabolites
        mets = list(value.keys())
        flag = False
        for met in mets:
            if met not in all_mets: 
                flag = True
                break
        if flag: continue

        rule_df[rid] = 0
        for met, stoic in list(value.items()):
            if met in mets_TECRDB:
                continue  # hydogen is zero
            rule_df[rid] += molsigna_df[met] * stoic
    rule_df.to_csv("./data/reaction_rule_nostereo_remove_TECRDB_mets.csv", index=True)



if __name__ == '__main__':
    # db = pd.read_csv('./data/cache_compounds_20160818.csv',index_col='compound_id')
    # db_smiles = db['smiles_pH7'].to_dict()
    # decompse_ac(db_smiles)
    # get_rxn_rule()

    # get_rxn_rule_remove_TECRDB_mets()
    get_rxn_rule_no_stero_remove_TECRDB_mets()