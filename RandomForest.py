from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef

import features_extractor

fe_train = features_extractor.FeatureExtractor("supplementary_small/")
fe_test = features_extractor.FeatureExtractor("supplementary_small_test/")


train_x = fe_train.features
train_y = fe_train.labels

test_x = fe_test.features
test_y = fe_test.labels


clf = RandomForestClassifier(n_estimators=100, max_depth=2, min_samples_leaf=1, min_samples_split=5, random_state=0)
clf.fit(train_x, train_y)


pred_train = list(clf.predict(train_x))
pred_test = list(clf.predict(test_x))

print(test_y)
print(pred_test)

#for f in clf.feature_importances_:
    #print(f)

print("Train Accuracy   : ", accuracy_score(train_y, pred_train))
print("Test Accuracy    : ", accuracy_score(test_y, pred_test))