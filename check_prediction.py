from matplotlib import pyplot as plt

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import os
import numpy as np
import pickle

from SOV import calculate_sov
from features_extractor import FeatureExtractor
from SOV import calculate_sov


def main():
    fe = FeatureExtractor('supplementary_test/')

    model_pickle = './best_models/KNN/knn_q3.pkl'

    with open(model_pickle, 'rb') as mod_file:
        model = pickle.load(mod_file)
        score = model.best_estimator_.score(fe.normalized_features, fe.labels_q3)
        print(score)
        prediction = model.best_estimator_.predict(fe.normalized_features)
        print(list(prediction))
        print(fe.labels_q3)

        print(calculate_sov(prediction, fe.labels_q3))

if __name__ == '__main__':
    main()
