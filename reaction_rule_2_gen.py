import pandas as pd
import pdb
import json
from rdkit import Chem

reaction_dict = json.load(open('./data/optstoic_v3_Sji_dict.json'))
molecular_signature = json.load(open('./data/decompose_vector_ac_r2_py3_indent_modified_manual.json'))
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
rule_df.to_csv("./data/reaction_rule_r2_py3_manual_modified.csv", index=True)
