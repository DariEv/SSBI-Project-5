from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import make_scorer
from features_extractor import FeatureExtractor
from SOV import calculate_sov

import pickle
import os
import numpy as np

# paths to input and output
FEATURES_FILE = 'Extracted_Features.pkl'
OUTPUT_GS_FILE_Q6 = './best_models/SVM/svm_q6.pkl'
OUTPUT_GS_FILE_Q3 = './best_models/SVM/svm_q3.pkl'


def main():

    # check if feature file exists
    if not os.path.isfile(FEATURES_FILE):
        print('The feature file cannot be loaded -> Exiting!')
        return

    # open saved file

    with open(FEATURES_FILE, 'rb') as input:
        saved_features = pickle.load(input)
        features = saved_features.normalized_features
        #features = saved_features.local_features
        q6 = saved_features.labels_q6
        q3 = saved_features.labels_q3

    X = np.array(features)
    y_q3 = np.array(q3)
    y_q6 = np.array(q6)

    # add custom scoring method SVO
    scoring = {'Accuracy': 'accuracy', 'SOV': make_scorer(calculate_sov, greater_is_better=True)}

    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}

    svc = svm.SVC(gamma="scale")
    clf = GridSearchCV(svc, parameters,
                       scoring=scoring,
                       cv=4,
                       refit='Accuracy',
                       return_train_score=True,
                       verbose=2)
    clf.fit(X, y_q6)

    # print results

    results = clf.cv_results_
    print()
    print('###############   Best CV Results\n')
    print(clf.best_estimator_)
    print(clf.best_score_)
    print(clf.best_params_)
    for metric in scoring.keys():
        print('Best mean test {}: {}'.format(metric, results['mean_test_{}'.format(metric)][clf.best_index_]))


    ############## Save q6 for later evaluation #######################

    with open(OUTPUT_GS_FILE_Q6, 'wb') as outputQ6:
        print('SAVE GRID SEARCH OBJECT Q6')
        pickle.dump(clf, outputQ6, pickle.HIGHEST_PROTOCOL)

    ###################################################################

    best_parameters = {}
    for k, v in clf.best_params_.items():
        best_parameters[k] = np.array([v])

    # Compare best model on q3 Task
    model_q3 = svm.SVC(gamma="scale")
    gs_q3 = GridSearchCV(model_q3,
                         param_grid=best_parameters,
                         scoring=scoring, cv=4,
                         refit='Accuracy',
                         return_train_score=True)

    gs_q3.fit(X, y_q3)
    results_q3 = gs_q3.cv_results_

    print()
    print('###############    Results of model on the q3 task\n')
    for metric in scoring.keys():
        print('Best mean test {}: {}'.format(metric, results_q3['mean_test_{}'.format(metric)][gs_q3.best_index_]))


    ############## Save q3 for later evaluation #######################

    with open(OUTPUT_GS_FILE_Q3, 'wb') as outputQ3:
        print('SAVE GRID SEARCH OBJECT Q3')
        pickle.dump(gs_q3, outputQ3, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    main()

    '''
    scoring = {'Accuracy': 'accuracy', 'SOV': make_scorer(calculate_sov, greater_is_better=True)}

    with open(OUTPUT_GS_FILE_Q3, 'rb') as inputQ3:
        q3 = pickle.load(inputQ3)
        results = q3.cv_results_

        print('###############   Q3: Best CV Results\n')
        print(q3.best_estimator_)
        print(q3.best_score_)
        print(q3.best_params_)
        for metric in scoring.keys():
            print('Best mean test {}: {}'.format(metric, results['mean_test_{}'.format(metric)][q3.best_index_]))

    print("----------------------------")

    with open(OUTPUT_GS_FILE_Q6, 'rb') as inputQ6:
        q6 = pickle.load(inputQ6)
        #print(q6.cv_results_)

        results_q6 = q6.cv_results_

        print('###############  Q6: Best CV Results\n')
        print(q6.best_estimator_)
        print(q6.best_score_)
        print(q6.best_params_)
        for metric in scoring.keys():
            print('Best mean test {}: {}'.format(metric, results_q6['mean_test_{}'.format(metric)][q6.best_index_]))
    '''
