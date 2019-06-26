import pickle
import numpy as np
from features_extractor import FeatureExtractor


# access to features file is required for getting peptide sequence lengths
FEATURES_FILE = 'Extracted_Features.pkl'


def calculate_sov(actual_labels, predicted_labels):

    # open saved peptide lengths

    with open(FEATURES_FILE, 'rb') as input:
        saved_features = pickle.load(input)
        lengths = saved_features.peptide_lengths

    # iterate through peptides, extract overlapped segments

    i = 0
    sovs = []
    total_length = 0

    for l in lengths:

        # get labels for current peptide
        predicted_per_peptide = predicted_labels[i:l]
        actual_per_peptide = actual_labels[i:l]

        segments, segments_length = get_overlapped_segments(predicted_per_peptide, actual_per_peptide)

        # concat lists of overlapped segments
        sovs = sovs + segments
        total_length = total_length + segments_length

        i = l

    # calculate resulting score
    sov = score_sov(sovs, total_length)

    return sov


def get_overlapped_segments(actual_labels, predicted_labels):

    segments = []
    total_length = 0

    current_label = np.inf # label of previously detected overlapped segment
    n = len(actual_labels)

    # iterate through actual labels
    for i, label_actual in enumerate(actual_labels):

        # we are in the already detected overlap
        if label_actual == current_label:
            continue

        # detected overlap is finished
        if label_actual != current_label:
            current_label = np.inf

        # overlap is found
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

            # save all needed info
            overlapped_pair = {"label": current_label,
                               "union": union_length,
                               "intersection": intersection_length,
                               "l_actual": l_actual,
                               "l_predicted": l_predicted}

            segments.append(overlapped_pair)
            total_length = total_length + l_predicted

    return segments, total_length


# apply formula from the lecture

def score_sov(sovs, total_length):

    score = 0

    for o_lap_pair in sovs:
        score = score + ((o_lap_pair['intersection'] + delta(o_lap_pair)) / o_lap_pair['union']) * o_lap_pair['l_predicted']

    return score / total_length


def delta(segment_pair):

    return min([segment_pair['union'] - segment_pair['intersection'],
                segment_pair['intersection'],
                segment_pair['l_actual'] / 2,
                segment_pair['l_predicted'] / 2])


def main():

    # FOR TESTING

    # open saved file

    '''
    with open(FEATURES_FILE, 'rb') as input:
        saved_features = pickle.load(input)
        features = saved_features.normalized_features
        q6 = saved_features.labels_q6
        q3 = saved_features.labels_q3
        lengths = saved_features.peptide_lengths


    #print(lengths)
    #print(q3)
    #print(q6)

    sov = calculate_sov(q3, q3)
    print(sov)
    '''


    # test CV
    '''
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer

    X = features
    y = q6

    svo_scorer = make_scorer(calculate_sov, greater_is_better=True)

    print(cross_val_score(DummyClassifier(), X, y,  scoring=svo_scorer, cv=5))
    '''


if __name__ == '__main__':
    main()
