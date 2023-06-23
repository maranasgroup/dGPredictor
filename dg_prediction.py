import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import webbrowser
import pickle
import joblib

from CC.chemaxon import *
from CC.compound import Compound
from CC.compound_cacher import CompoundCacher
from rdkit.Chem import rdChemReactions as Reactions
from rdkit.Chem import Draw
from rdkit import Chem

import json, sys, re, os

class dGPredictor:

    def __init__(self, smiles_compounds_path=None, mol_sig_r1_path=None,
                 mol_sig_r2_path=None, model_file_path=None):
        # load_smiles
        smiles_compounds_path = smiles_compounds_path or os.path.join(
            os.path.dirname(__file__), 'data/cache_compounds_20160818.csv')
        db = pd.read_csv(smiles_compounds_path, index_col='compound_id')
        self.db_smiles = db['smiles_pH7'].to_dict()

        # load_molsig_rad1
        mol_sig_r1_path = mol_sig_r1_path or os.path.join(
            os.path.dirname(__file__), 'data/decompose_vector_ac.json')
        self.mol_sig_r1 = json.load(open(mol_sig_r1_path))

        # load_molsig_rad2
        mol_sig_r2_path = mol_sig_r2_path or os.path.join(
            os.path.dirname(__file__), 'data/decompose_vector_ac_r2_py3_indent_modified_manual.json')
        self.mol_sig_r2 = json.load(open(mol_sig_r2_path))

        # load_model
        model_file_path = model_file_path or os.path.join(os.path.dirname(__file__), 'model/M12_model_BR.pkl')
        self.model = joblib.load(open(model_file_path, 'rb'))

        # load_compound_cache
        self.ccache = CompoundCacher()

    def predict(self, rxn_str, rxnID, pH, I, extra_info=None, draw=True, printing=True):
        # parameterize novel contributions
        novel_mets = dGPredictor.parse_novel_molecule(extra_info)
        novel_smiles = dGPredictor.parse_novel_smiles(novel_mets)
        novel_decomposed_r1 = dGPredictor.decompse_novel_mets_rad1(novel_smiles)
        novel_decomposed_r2 = dGPredictor.decompse_novel_mets_rad2(novel_smiles)

        # draw the simulated reaction
        rxn_dict = dGPredictor.parse_formula(rxn_str)
        if draw:
            dGPredictor.draw_rxn_figure(rxn_dict, self.db_smiles, novel_smiles)

        # estimate the dG for the reaction
        mu, std, rule_df1, rule_df2 = dGPredictor.get_dG0(
            rxn_dict, rxnID, pH, I, self.model, self.molsig_r1, self.molsig_r2,
            novel_decomposed_r1, novel_decomposed_r2, novel_mets)
        if printing:
            print(f"{rxnID}:\tdG = {mu:.2f} ± {std:.2f} kJ/mol")
        return mu, std, rule_df1, rule_df2

    def bulk_prediction(self, RXNs, pH, I, extra_info=None, draw=True, printing=True):
        return {rxnID: self.predict(rxn_str, rxnID, pH, I, extra_info, draw, printing)
                for rxnID, rxn_str in RXNs.items()}

    @staticmethod
    def count_substructures(radius, molecule):
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
            env = Chem.FindAtomEnvironmentOfRadiusN(m, radius, i)
            atoms = set()
            for bidx in env:
                atoms.add(m.GetBondWithIdx(bidx).GetBeginAtomIdx())
                atoms.add(m.GetBondWithIdx(bidx).GetEndAtomIdx())

            # only one atom is in this environment, such as O in H2O
            if len(atoms) == 0:
                atoms = {i}

            smi = Chem.MolFragmentToSmiles(m, atomsToUse=list(atoms),
                                           bondsToUse=env, canonical=True)

            if smi in smi_count:
                smi_count[smi] = smi_count[smi] + 1
            else:
                smi_count[smi] = 1
        return smi_count

    @staticmethod
    def decompse_novel_mets_rad1(novel_smiles, radius=1):
        decompose_vector = dict()

        for cid, smiles_pH7 in novel_smiles.items():
            mol = Chem.MolFromSmiles(smiles_pH7)
            mol = Chem.RemoveHs(mol)
            # Chem.RemoveStereochemistry(mol)
            smi_count = dGPredictor.count_substructures(radius, mol)
            decompose_vector[cid] = smi_count
        return decompose_vector

    @staticmethod
    def decompse_novel_mets_rad2(novel_smiles, radius=2):
        decompose_vector = dict()

        for cid, smiles_pH7 in novel_smiles.items():
            mol = Chem.MolFromSmiles(smiles_pH7)
            mol = Chem.RemoveHs(mol)
            # Chem.RemoveStereochemistry(mol)
            smi_count = dGPredictor.count_substructures(radius, mol)
            decompose_vector[cid] = smi_count
        return decompose_vector

    @staticmethod
    def parse_reaction_formula_side(s):
        """
            Parses the side formula, e.g. '2 C00001 + C00002 + 3 C00003'
            Ignores stoichiometry.

            Returns:
                The set of CIDs.
        """
        if s.strip() == "null":
            return {}

        compound_bag = {}
        for member in re.split('\s+\+\s+', s):
            tokens = member.split(None, 1)
            if len(tokens) == 0:
                continue
            if len(tokens) == 1:
                amount = 1
                key = member
            else:
                amount = float(tokens[0])
                key = tokens[1]

            compound_bag[key] = compound_bag.get(key, 0) + amount

        return compound_bag

    @staticmethod
    def parse_formula(formula, arrow='<=>', rid=None):
        """
            Parses a two-sided formula such as: 2 C00001 => C00002 + C00003

            Return:
                The set of substrates, products and the direction of the reaction
        """
        tokens = formula.split(arrow)
        if len(tokens) < 2:
            print(('Reaction does not contain the arrow sign (%s): %s'
                   % (arrow, formula)))
        if len(tokens) > 2:
            print(('Reaction contains more than one arrow sign (%s): %s'
                   % (arrow, formula)))

        left = tokens[0].strip()
        right = tokens[1].strip()

        sparse_reaction = {}
        for cid, count in dGPredictor.parse_reaction_formula_side(left).items():
            sparse_reaction[cid] = sparse_reaction.get(cid, 0) - count

        for cid, count in dGPredictor.parse_reaction_formula_side(right).items():
            sparse_reaction[cid] = sparse_reaction.get(cid, 0) + count

        return sparse_reaction

    @staticmethod
    def draw_rxn_figure(rxn_dict, db_smiles, novel_smiles):
        # db_smiles = load_smiles()

        left = ''
        right = ''

        for met, stoic in rxn_dict.items():
            if met == "C00080" or met == "C00282":
                continue  # hydogen is not considered
            if stoic > 0:
                if met in db_smiles:
                    right = right + db_smiles[met] + '.'
                else:
                    right = right + novel_smiles[met] + '.'
            else:
                if met in db_smiles:
                    left = left + db_smiles[met] + '.'
                else:
                    left = left + novel_smiles[met] + '.'
        smarts = left[:-1] + '>>' + right[:-1]
        # print smarts
        smarts = str(smarts)
        rxn = Reactions.ReactionFromSmarts(smarts, useSmiles=True)
        return Chem.Draw.ReactionToImage(rxn)  # , subImgSize=(400, 400))

    @staticmethod
    def get_rule(rxn_dict, molsig1, molsig2, novel_decomposed1, novel_decomposed2):
        if novel_decomposed1 != None:
            for cid in novel_decomposed1:
                molsig1[cid] = novel_decomposed1[cid]
        if novel_decomposed2 != None:
            for cid in novel_decomposed2:
                molsig2[cid] = novel_decomposed2[cid]

        molsigna_df1 = pd.DataFrame.from_dict(molsig1).fillna(0)
        all_mets1 = molsigna_df1.columns.tolist()
        all_mets1.append("C00080")
        all_mets1.append("C00282")

        molsigna_df2 = pd.DataFrame.from_dict(molsig2).fillna(0)
        all_mets2 = molsigna_df2.columns.tolist()
        all_mets2.append("C00080")
        all_mets2.append("C00282")

        moieties_r1 = open(os.path.join(os.path.dirname(__file__), 'data/group_names_r1.txt'))
        moieties_r2 = open(os.path.join(os.path.dirname(__file__), 'data/group_names_r2_py3_modified_manual.txt'))
        moie_r1 = moieties_r1.read().splitlines()
        moie_r2 = moieties_r2.read().splitlines()

        molsigna_df1 = molsigna_df1.reindex(moie_r1)
        molsigna_df2 = molsigna_df2.reindex(moie_r2)

        rule_df1 = pd.DataFrame(index=molsigna_df1.index)
        rule_df2 = pd.DataFrame(index=molsigna_df2.index)
        # for rid, value in reaction_dict.items():
        #     # skip the reactions with missing metabolites
        #     mets = value.keys()
        #     flag = False
        #     for met in mets:
        #         if met not in all_mets:
        #             flag = True
        #             break
        #     if flag: continue

        rule_df1['change'] = 0
        for met, stoic in rxn_dict.items():
            if met == "C00080" or met == "C00282":
                continue  # hydogen is zero
            rule_df1['change'] += molsigna_df1[met] * stoic

        rule_df2['change'] = 0
        for met, stoic in rxn_dict.items():
            if met == "C00080" or met == "C00282":
                continue  # hydogen is zero
            rule_df2['change'] += molsigna_df2[met] * stoic

        rule_vec1 = rule_df1.to_numpy().T
        rule_vec2 = rule_df2.to_numpy().T

        m1, n1 = rule_vec1.shape
        m2, n2 = rule_vec2.shape

        zeros1 = np.zeros((m1, 44))
        zeros2 = np.zeros((m2, 44))
        X1 = np.concatenate((rule_vec1, zeros1), 1)
        X2 = np.concatenate((rule_vec2, zeros2), 1)

        rule_comb = np.concatenate((X1, X2), 1)

        # rule_df_final = {}
        # rule_df_final['rad1'] = rule_df1
        # rule_df_final['rad2'] = rule_df2
        return rule_comb, rule_df1, rule_df2

    @staticmethod
    def get_ddG0(rxn_dict, pH, I, novel_mets):
        ccache = CompoundCacher()
        # ddG0 = get_transform_ddG0(rxn_dict, ccache, pH, I, T)
        T = 298.15
        ddG0_forward = 0
        for compound_id, coeff in rxn_dict.items():
            if novel_mets != None and compound_id in novel_mets:
                comp = novel_mets[compound_id]
            else:
                comp = ccache.get_compound(compound_id)
            ddG0_forward += coeff * comp.transform_pH7(pH, I, T)

        return ddG0_forward

    @staticmethod
    def get_dG0(rxn_dict, rid, pH, I, loaded_model, molsig_r1, molsig_r2, novel_decomposed_r1, novel_decomposed_r2,
                novel_mets):

        # rule_df = get_rxn_rule(rid)
        rule_comb, rule_df1, rule_df2 = dGPredictor.get_rule(
            rxn_dict, molsig_r1, molsig_r2, novel_decomposed_r1, novel_decomposed_r2)

        X = rule_comb

        ymean, ystd = loaded_model.predict(X, return_std=True)

        result = {}
        # result['dG0'] = ymean[0] + get_ddG0(rxn_dict, pH, I)
        # result['standard deviation'] = ystd[0]

        # result_df = pd.DataFrame([result])
        # result_df.style.hide_index()
        # return result_df
        return ymean[0] + dGPredictor.get_ddG0(rxn_dict, pH, I, novel_mets), ystd[0], rule_df1, rule_df2
        # return ymean[0],ystd[0]

    @staticmethod
    def parse_novel_molecule(add_info):
        result = {}
        for cid, InChI in add_info.items():
            c = Compound.from_inchi('Test', cid, InChI)
            result[cid] = c
        return result

    @staticmethod
    def parse_novel_smiles(result):
        novel_smiles = {}
        for cid, c in result.items():
            smiles = c.smiles_pH7
            novel_smiles[cid] = smiles
        return novel_smiles