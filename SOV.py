import pickle
import collections
import numpy as np
from features_extractor import FeatureExtractor

Q3 = [0, 1, 2]
Q6 = [0, 1, 2, 3, 4, 5]

FEATURES_FILE = 'Extracted_Features_small.pkl'


def calculate_sov(actual_labels, predicted_labels):

    # get q type

    if 3 in predicted_labels or 4 in predicted_labels or 5 in predicted_labels:
        q = Q6
    else:
        q = Q3
    print(q)

    # open saved peptide lengths

    with open(FEATURES_FILE, 'rb') as input:
        saved_features = pickle.load(input)
        lengths = saved_features.peptide_lengths

    # iterate through peptides

    i = 0
    sovs = []
    for l in lengths:
        predicted_per_peptide = predicted_labels[i:l]
        actual_per_peptide = actual_labels[i:l]
        sovs = sovs + get_overlapped_segments(predicted_per_peptide, actual_per_peptide)
        i = l

    print(sovs)

    sov = 0

    return 0.5


def get_overlapped_segments(actual_labels, predicted_labels):

    segments = []

    current_label = np.inf
    n = len(actual_labels)

    for i, label_actual in enumerate(actual_labels):

        if label_actual == current_label:
            continue

        if label_actual != current_label:
            current_label = np.inf

        if label_actual == predicted_labels[i]:

            current_label = label_actual
            intersection_length = 1
            union_length = 1
            l_actual = 1
            l_predicted = 1

            # iterate to the left
            k = i - 1
            while k >= 0 and (actual_labels[k] == current_label or predicted_labels[k] == current_label):
                union_length = union_length + 1
                if actual_labels[k] == predicted_labels[k]:
                    intersection_length = intersection_length + 1
                    l_actual = l_actual + 1
                    l_predicted = l_predicted + 1
                elif actual_labels[k] == current_label:
                    l_actual = l_actual + 1
                elif predicted_labels[k] == current_label:
                    l_predicted = l_predicted + 1
                k = k - 1

            # iterate to the right
            k = i + 1
            while k < n and (actual_labels[k] == current_label or predicted_labels[k] == current_label):
                union_length = union_length + 1
                if actual_labels[k] == predicted_labels[k]:
                    intersection_length = intersection_length + 1
                    l_actual = l_actual + 1
                    l_predicted = l_predicted + 1
                elif actual_labels[k] == current_label:
                    l_actual = l_actual + 1
                elif predicted_labels[k] == current_label:
                    l_predicted = l_predicted + 1
                k = k + 1

            overlapped_pair = {"label": current_label,
                               "union": union_length,
                               "intersection": intersection_length,
                               "l_actual": l_actual,
                               "l_predicted": l_predicted}

            segments.append(overlapped_pair)

    return segments


def main():

    # open saved file

    with open(FEATURES_FILE, 'rb') as input:
        saved_features = pickle.load(input)
        q6 = saved_features.labels_q6
        q3 = saved_features.labels_q3
        lengths = saved_features.peptide_lengths

    #calculate_sov(q3, q3)

    #print(lengths)
    print(q3)
    print(q6)

    sovs = calculate_sov(q3, q6)
    #print(sovs)


    #### test CV
    '''
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer

    X = [[1], [1], [1], [1], [1],
         [1], [1], [1], [1], [1],
         [1], [1], [1], [1], [1]]
    y = [0, 1, 0, 1, 1,
         0, 1, 1, 0, 1,
         1, 0, 1, 1, 0]

    svo_scorer = make_scorer(calculate_sov, greater_is_better=False)

    print(cross_val_score(DummyClassifier(), X, y,  scoring=svo_scorer, cv=5))
    '''


if __name__ == '__main__':
    main()
