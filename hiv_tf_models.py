"""
Script that trains multitask models on hiv dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import deepchem as dc
import numpy as np
from deepchem.molnet import load_hiv, featurizers
from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer
from utils import *

# Only for debug!
np.random.seed(123)

# Load hiv dataset
n_features = 1024
hiv_tasks, hiv_datasets, transformers = load_hiv()
train_dataset, valid_dataset, test_dataset = hiv_datasets

test_smiles = 'c1ccccc1'
featurizer = dc.feat.CircularFingerprint(size=1024)
ecfp = featurizer.featurize(test_smiles)

smiles_dataset = dc.data.NumpyDataset(X=ecfp)
print(f'Size of ECFP: {ecfp.shape}')

# Fit models
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

model = dc.models.MultitaskClassifier(
    len(hiv_tasks),
    n_features,
    layer_sizes=[1000],
    dropouts=[.25],
    learning_rate=0.001,
    batch_size=50)  # batch_size default 50

# Fit trained model
model.fit(train_dataset)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

rand_arr = np.random.rand(50, 1024)
rand_data = dc.data.DiskDataset.from_numpy(X=rand_arr)

print('-'*20)
print(model.predict(smiles_dataset))

print(model.save_checkpoint(1, 'checkpoints'))

