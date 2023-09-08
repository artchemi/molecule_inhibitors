from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import numpy as np
import deepchem as dc
import os
from deepchem.molnet import load_hiv, featurizers
from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer
from utils import *


def main():
    smiles_lst = txt_parser('test_smiles.txt')  # Собираем список SMILES из файла
    featurizer = dc.feat.CircularFingerprint(size=1024)  # Фичи
    ecfp = featurizer.featurize(smiles_lst)

    smiles_dataset = dc.data.NumpyDataset(X=ecfp)  # Датасет

    model = dc.models.MultitaskClassifier(
        1,
        1024,
        layer_sizes=[1000],
        dropouts=[.25],
        learning_rate=0.001,
        batch_size=50)  # batch_size default 50

    set_checkpoint(model, 'hiv')  # Устанавливаем чекпоинты

    print(model.predict(smiles_dataset))


if __name__ == '__main__':
    main()
