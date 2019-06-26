'''
This file is used to evaluate saved models on unseen test data.
'''

import pickle

from SOV import calculate_sov
from features_extractor import FeatureExtractor
from SOV import calculate_sov


def main():
    # Create the Features from the given test set
    fe = FeatureExtractor('supplementary_test/')

    # Path to the pickle file that contains the model we want to evaluate
    model_pickle = './best_models/KNN/knn_q3.pkl'

    # open the model file
    with open(model_pickle, 'rb') as mod_file:
        # load the model
        model = pickle.load(mod_file)
        # get the accuracy on the unseen data
        score = model.best_estimator_.score(fe.normalized_features, fe.labels_q3)
        print('Accuracy: ',score)
        # get the actual prediction of the unseen data
        prediction = model.best_estimator_.predict(fe.normalized_features)
        # print and calculate the SOV score
        print('SOV: ', calculate_sov(prediction, fe.labels_q3))
        print('Predictions:\n')
        print('Predicted labels: ',list(prediction))
        print('True labels: ',fe.labels_q3)


if __name__ == '__main__':
    main()
