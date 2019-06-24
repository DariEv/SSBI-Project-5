from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

import pickle
from features_extractor import FeatureExtractor
import SOV

FEATURES_FILE = 'Extracted_Features.pkl'


def main():
    #fe_train = features_extractor.FeatureExtractor("supplementary_small/")
    #fe_test = features_extractor.FeatureExtractor("supplementary_small_test/")

    # open saved file

    with open(FEATURES_FILE, 'rb') as input:
        saved_features = pickle.load(input)
        features = saved_features.normalized_features
        q6 = saved_features.labels_q6
        q3 = saved_features.labels_q3
        lengths = saved_features.peptide_lengths

    svo_scorer = make_scorer(SOV.calculate_sov, greater_is_better=True)
    #print(cross_val_score(LinearSVC(), features, q3, scoring=svo_scorer, cv=5))

    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svc = svm.SVC(gamma="scale")
    clf = GridSearchCV(svc, parameters, scoring=svo_scorer, cv=5)
    clf.fit(features, q3)
    print(clf.best_estimator_, clf.best_score_, clf.best_params_)

    print(clf.cv_results_)

    '''
    train_x = fe_train.features
    train_y = fe_train.labels
    
    test_x = fe_test.features
    test_y = fe_test.labels
    
    
    clf = LinearSVC(random_state=0, tol=1e-5, multi_class='ovr')
    clf.fit(train_x, train_y)
    
    pred_train = list(clf.predict(train_x))
    pred_test = list(clf.predict(test_x))
    
    print(train_y)
    print(pred_train)
    
    print("Train Accuracy   : ", accuracy_score(train_y, pred_train))
    print("Test Accuracy    : ", accuracy_score(test_y, pred_test))
    '''




if __name__ == '__main__':
    main()