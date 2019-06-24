from matplotlib import pyplot as plt

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import os
import numpy as np
import pickle

from SOV import calculate_sov
from features_extractor import FeatureExtractor

#FEATURES_FILE = 'Extracted_Features.pkl'
FEATURES_FILE = 'Extracted_Features_small.pkl'
OUTPUT_GS_FILE_Q6 = './best_models/KNN/knn_q6.pkl'
OUTPUT_GS_FILE_Q3 = './best_models/KNN/knn_q3.pkl'

def main():

    if not os.path.isfile(FEATURES_FILE):
        print('The feature file cannot be loaded -> Exiting!')
        return

    with open(FEATURES_FILE, 'rb') as input:
        saved_features = pickle.load(input)
        features = saved_features.normalized_features
        q6 = saved_features.labels_q6
        q3 = saved_features.labels_q3
        lengths = saved_features.peptide_lengths

    scoring = {'Accuracy': 'accuracy', 'SOV': make_scorer(calculate_sov, greater_is_better=True)}

    X = np.array(features)
    y_q6 = np.array(q6)
    y_q3 = np.array(q3)

    model = KNeighborsClassifier(algorithm='kd_tree')

    gs = GridSearchCV(model,
                      param_grid={'n_neighbors': range(5,16,5),
                                  'leaf_size': range(10,31,10)
                                  },
                      scoring=scoring, cv=4,
                      refit='Accuracy',
                      return_train_score=True,
                      verbose=2)

    gs.fit(X,y_q6)
    results = gs.cv_results_

    print()
    print('###############   Best CV Results\n')
    print(gs.best_estimator_)
    print(gs.best_score_)
    print(gs.best_params_)
    for metric in scoring.keys():
        print('Best mean test {}: {}'.format(metric, results['mean_test_{}'.format(metric)][gs.best_index_]))

    ############## Save q6 for later evaluation #######################
    with open(OUTPUT_GS_FILE_Q6, 'wb') as output_q6:
        print('SAVE GRID SEARCH OBJECT Q6')
        pickle.dump(gs, output_q6, pickle.HIGHEST_PROTOCOL)

    ###################################################################

    best_parameters = {}
    for k, v in gs.best_params_.items():
        best_parameters[k] = np.array([v])

    # Compare best model on q3 Task
    model_q3 = KNeighborsClassifier(algorithm='kd_tree')
    gs_q3 = GridSearchCV(model_q3,
                         param_grid=best_parameters,
                         scoring=scoring, cv=4,
                         refit='Accuracy',
                         return_train_score=True,
                         verbose=2)

    gs_q3.fit(X,y_q3)
    results_q3 = gs_q3.cv_results_

    print()
    print('###############    Results of model on the q3 task\n')
    for metric in scoring.keys():
        print('Best mean test {}: {}'.format(metric, results_q3['mean_test_{}'.format(metric)][gs_q3.best_index_]))

    ############## Save q6 for later evaluation #######################
    with open(OUTPUT_GS_FILE_Q3, 'wb') as output_q3:
        print('SAVE GRID SEARCH OBJECT Q3')
        pickle.dump(gs_q3, output_q3, pickle.HIGHEST_PROTOCOL)

    ###################################################################


if __name__ == '__main__':
    main()

