from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, matthews_corrcoef


import features_extractor

fe_train = features_extractor.FeatureExtractor("supplementary_small/")
fe_test = features_extractor.FeatureExtractor("supplementary_small_test/")


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




