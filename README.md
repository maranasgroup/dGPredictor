# dGPredictor

==================================
### Requirements:

1. Python 3.8.10
2. RDkit (http://www.rdkit.org/)
3. pandas (https://pandas.pydata.org/)
4. matplotlib (https://matplotlib.org/stable/users/installing.html)
5. Scikit-learn (https://scikit-learn.org/stable/)
6. Streamlit (https://streamlit.io/)
7. Openbabel (https://anaconda.org/openbabel/openbabel)
8. ChemAxon's Marvin >= 5.11 
9. Pulp

Installation
1. Python 3.8.10 (https://www.python.org/downloads/windows/)
Recommended- 
- Create anaconda environment using command "conda create -n dGPredictor python=3.8 ipython"
- activate the env using command "conda activate dGPredictor" or "source activate dGPredictor"
2. RDkit
- type command "conda install -c conda-forge rdkit" in your dGPredictor env to install rdkit
3. Pandas
- "conda install pandas"
4. matplotlib
- "conda install -c conda-forge matplotlib"
5. Scikit-learn
- use command "pip install -U scikit-learn"
6. Streamlit 
- use command "pip install -U streamlit"
7. Openbabel
- run "conda install -c conda-forge openbabel" 
8. ChemAxon's Marvin (PkA value estimation)
- Marvin is only required for adding structures of novel metabolites/compounds that are not in the KEGG database
- instructions (https://chemaxon.com/products/marvin/download)
- add "cxcalc.bat (macOS) /cxcalc.exe (Windows)" to PATH and also in "./CC/chemaxon.py" file
- you will need to get a license to use ChemAxon (it is free for academic use)
9. Pulp
- use command "pip install -U pulp"




==================================
### Running web-interface locally using streamlit

- Model generation: Run "model_gen.py" using "python model_gen.py" once to create dGPredictor model file :- (Running this might take some time)
- run "streamlit run ./streamlit/main.py" from dGPredictor folder
- running KEGG reaction (doesn't require ChemAxon's Marvin) : copy paste the reaction equation into reaction section and click search

### Gibbs free energy prediction use automated group decomposition method

- Step 1: decompose the metabolites based on smiles files (see function decompse_ac in decompose_groups.py or notebook )
- Step 2: create group changes vectors (i.e. reaction rules) based on group changes in metabolites of reactions (see get_rxn_rule in decompose_groups.py)
- Step 3: linear regression, Ridge Regression and Bayesian Ridge Regression in "predict.py"
- Step 4: Multiple regression models in notebook "analysis_dGPredictor.ipynb"

### Pathway design using novoStoic
- Run "mini_novoStoic.py" to see an example to design pathways for Isobutanol synthesis


# demo
![dGPredictor Demo](figures/dg_demo_py3.gif)

