"""Module for runnging the whole model."""

from widedeepdta import WideDeepDTA
from evaluation import evaluate_predictions
import numpy as np
import pandas as pd
import json
import os
# %%
def get_embeddings(embedding_path, emb_dim, len_set, bdb_data):
    """
    Retrieve the embeddings of given molecules according to BDB ordering.

    Parameters
    ----------
    embedding_path : str
        Path to the embedding's file.
    emb_dim : int
        Embedding imension.
    len_set : int
        Length of the requested set.
    bdb_data : list
        Lits of sequences.

    Returns
    -------
    loaded_embeddings : Array of float64
        Embeddings of corresponding molecules.

    """
    if(os.path.exists(embedding_path)): 
        with open(embedding_path) as fd:
            embeddings = json.load(fd)
        loaded_embeddings = []
        for cid in bdb_data:
            if str(cid) in embeddings:
                loaded_embeddings.append(embeddings[str(cid)])
            else:
                loaded_embeddings.append([0.0] * emb_dim)
        loaded_embeddings = np.asarray(loaded_embeddings, dtype="float64")
    else:
        loaded_embeddings = np.array([[float(0)] * emb_dim for _ in range(len_set)]).astype(np.float64)
    return loaded_embeddings
# %%
def retrieve_bdb_setup(bdb_folds_path, ix):
    """
    Return requested BDB data fold from given path.

    Parameters
    ----------
    bdb_folds_path : str
        Path to the requested BDB data fold.
    ix : int
        Index of the setup.

    Returns
    -------
    bdb : dict
        Dictionary containing each fold of the selected setup.

    """
    bdb = {}
    bdb['train'] = pd.read_csv(bdb_folds_path + f'setup_{ix}/train.csv')
    bdb['cold_both'] = pd.read_csv(bdb_folds_path + f'setup_{ix}/test_cold_both.csv')
    bdb['cold_lig'] = pd.read_csv(bdb_folds_path + f'setup_{ix}/test_cold_lig.csv')
    bdb['cold_prot'] = pd.read_csv(bdb_folds_path + f'setup_{ix}/test_cold_prot.csv')
    bdb['warm'] = pd.read_csv(bdb_folds_path + f'setup_{ix}/test_warm.csv')
    return bdb 
# %%

model_name = 'Model14'
bdb_folds_path = '../data/bdb/setups/'
emb_dim = 32
chemical_embedding_path = f'../data/embedding/pubchem_drug_embeddings_{model_name}.json'
protein_embedding_path = f'../data/embedding/uniprot_protein_embeddings_{model_name}.json'

for ix in range(5):
    predictions = {}
    scores = {}
    savedir = f'../model/{model_name}/setup_{ix}/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    bdb = retrieve_bdb_setup(bdb_folds_path, ix)
    
    train_chemical_embeddings = get_embeddings(chemical_embedding_path, emb_dim, len_set=len(bdb['train']), bdb_data=bdb['train']['ligand_id'].tolist())
    train_protein_embeddings = get_embeddings(protein_embedding_path, emb_dim, len_set=len(bdb['train']), bdb_data=bdb['train']['prot_id'].tolist())
    
    cold_both_chemical_embeddings = get_embeddings(chemical_embedding_path, emb_dim, len_set=len(bdb['cold_both']), bdb_data=bdb['cold_both']['ligand_id'].tolist())
    cold_both_protein_embeddings = get_embeddings(protein_embedding_path, emb_dim, len_set=len(bdb['cold_both']), bdb_data=bdb['cold_both']['prot_id'].tolist())

    cold_lig_chemical_embeddings = get_embeddings(chemical_embedding_path, emb_dim, len_set=len(bdb['cold_lig']), bdb_data=bdb['cold_lig']['ligand_id'].tolist())
    cold_lig_protein_embeddings = get_embeddings(protein_embedding_path, emb_dim, len_set=len(bdb['cold_lig']), bdb_data=bdb['cold_lig']['prot_id'].tolist())

    cold_prot_chemical_embeddings = get_embeddings(chemical_embedding_path, emb_dim, len_set=len(bdb['cold_prot']), bdb_data=bdb['cold_prot']['ligand_id'].tolist())
    cold_prot_protein_embeddings = get_embeddings(protein_embedding_path, emb_dim, len_set=len(bdb['cold_prot']), bdb_data=bdb['cold_prot']['prot_id'].tolist())

    warm_chemical_embeddings = get_embeddings(chemical_embedding_path, emb_dim, len_set=len(bdb['warm']), bdb_data=bdb['warm']['ligand_id'].tolist())
    warm_protein_embeddings = get_embeddings(protein_embedding_path, emb_dim, len_set=len(bdb['warm']), bdb_data=bdb['warm']['prot_id'].tolist())
    
    model = WideDeepDTA(n_epochs=100)

    history = model.train(bdb['train']['smiles'].tolist(),
                          train_chemical_embeddings,
                          bdb['train']['aa_sequence'].tolist(),
                          train_protein_embeddings,
                          bdb['train']['affinity_score'].tolist())

    model.save(f'../model/{model_name}/setup_{ix}')
    
    predictions['train_predictions'] = model.predict(bdb['train']['smiles'].tolist(), train_chemical_embeddings, bdb['train']['aa_sequence'].tolist(), train_protein_embeddings)
    predictions['cold_both_predictions'] = model.predict(bdb['cold_both']['smiles'].tolist(), cold_both_chemical_embeddings, bdb['cold_both']['aa_sequence'].tolist(), cold_both_protein_embeddings)
    predictions['cold_lig_predictions'] = model.predict(bdb['cold_lig']['smiles'].tolist(), cold_lig_chemical_embeddings, bdb['cold_lig']['aa_sequence'].tolist(), cold_lig_protein_embeddings)
    predictions['cold_prot_predictions'] = model.predict(bdb['cold_prot']['smiles'].tolist(), cold_prot_chemical_embeddings, bdb['cold_prot']['aa_sequence'].tolist(), cold_prot_protein_embeddings)
    predictions['warm_predictions'] = model.predict(bdb['warm']['smiles'].tolist(), warm_chemical_embeddings, bdb['warm']['aa_sequence'].tolist(), warm_protein_embeddings)

    scores['train_scores'] = evaluate_predictions(bdb['train']['affinity_score'].tolist(), predictions['train_predictions'], ['ci', 'r2', 'rmse', 'mse'])
    scores['cold_both_scores'] = evaluate_predictions(bdb['cold_both']['affinity_score'].tolist(), predictions['cold_both_predictions'], ['ci', 'r2', 'rmse', 'mse'])
    scores['cold_lig_scores'] = evaluate_predictions(bdb['cold_lig']['affinity_score'].tolist(), predictions['cold_lig_predictions'], ['ci', 'r2', 'rmse', 'mse'])
    scores['cold_prot_scores'] = evaluate_predictions(bdb['cold_prot']['affinity_score'].tolist(), predictions['cold_prot_predictions'], ['ci', 'r2', 'rmse', 'mse'])
    scores['warm_scores'] = evaluate_predictions(bdb['warm']['affinity_score'].tolist(), predictions['warm_predictions'], ['ci', 'r2', 'rmse', 'mse'])

    with open(f'../model/{model_name}/setup_{ix}/scores.json', 'w') as f:
        json.dump(scores, f, indent=4)

    with open(f'../model/{model_name}/setup_{ix}/history.json', 'w') as f:
        json.dump(history, f, indent=4)

    with open(f'../model/{model_name}/setup_{ix}/predictions.json', 'w') as f:
        json.dump(predictions, f, indent=4)