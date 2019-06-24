from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

import os
import numpy as np
import pickle
from SOV import calculate_sov
from features_extractor import FeatureExtractor


FEATURES_FILE = 'Extracted_Features_small.pkl'
OUTPUT_GS_FILE_Q6 = './best_models/RF/rf_q6.pkl'
OUTPUT_GS_FILE_Q3 = './best_models/RF/rf_q3.pkl'


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

    # Model for q6 classification
    estimator_q6 = RandomForestClassifier()

    # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]  # 1000 - 0.83, 2000 - 0.84
    n_estimators = [100, 50]
    # Number of features to consider at every split
    max_features = [68, 10]
    # Method of selecting samples for training each tree
    #bootstrap = [True, False]
    # criterion for calculating the decision trees
    criterion = ['gini', 'entropy']

    param_grid = dict(n_estimators=n_estimators, max_features=max_features, criterion=criterion)
    print(param_grid)

    gs_q6 = GridSearchCV(estimator=estimator_q6, param_grid=param_grid,
                      scoring=scoring, cv=4,
                      refit='Accuracy',
                      return_train_score=True,
                      verbose=2)

    ##

    # model = RandomForestClassifier(n_estimators=100, max_depth=2, min_samples_leaf=1, min_samples_split=5, random_state=0)

    gs_q6.fit(X, y_q6)
    results = gs_q6.cv_results_

    # Get the feature importances, based on best estimator
    print()
    print('Feature importances: (index, value)')
    for i, f in enumerate(gs_q6.best_estimator_.feature_importances_):
        print(i, f)

    print()
    print('###############   Best CV Results\n')
    print(gs_q6.best_estimator_)
    print(gs_q6.best_score_)
    print(gs_q6.best_params_)
    for metric in scoring.keys():
        print('Best mean test {}: {}'.format(metric, results['mean_test_{}'.format(metric)][gs_q6.best_index_]))

    with open(OUTPUT_GS_FILE_Q6, 'wb') as output_q6:
        print('SAVE GRID SEARCH OBJECT Q6')
        pickle.dump(gs_q6, output_q6, pickle.HIGHEST_PROTOCOL)

    ###################################################################

    best_parameters = {}
    for k, v in gs_q6.best_params_.items():
        best_parameters[k] = np.array([v])


    # Compare best model on q3 task
    # Model for q3 classification
    estimator_q3 = RandomForestClassifier()

    # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]  # 1000 - 0.83, 2000 - 0.84
    n_estimators = [100, 50]
    # Number of features to consider at every split
    max_features = [68, 10]
    # Method of selecting samples for training each tree
    #bootstrap = [True, False]
    # criterion for calculating the decision trees
    criterion = ['gini', 'entropy']

    param_grid = dict(n_estimators=n_estimators, max_features=max_features, criterion=criterion)
    print(param_grid)

    gs_q3 = GridSearchCV(estimator=estimator_q3, param_grid=param_grid,
                         scoring=scoring, cv=4,
                         refit='Accuracy',
                         return_train_score=True,
                         verbose=2)

    gs_q3.fit(X, y_q3)
    results_q3 = gs_q3.cv_results_

    print()
    print('###############    Results of model on the q3 task\n')
    for metric in scoring.keys():
        print('Best mean test {}: {}'.format(metric, results_q3['mean_test_{}'.format(metric)][gs_q3.best_index_]))

    #Save q6 for later evaluation
    with open(OUTPUT_GS_FILE_Q3, 'wb') as output_q3:
        print('SAVE GRID SEARCH OBJECT Q3')
        pickle.dump(gs_q3, output_q3, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
