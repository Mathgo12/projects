import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml as fo
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.model_selection import cross_val_score
from sklearn import metrics

mnist = fo('mnist_784', version=1)
X,y = mnist['data'], mnist['target']
y = y.astype('uint8')
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train5 = (y_train == 5)
y_test5 = (y_test == 5)

classifier = SGD(random_state=54)
classifier.fit(X_train, y_train5)

test_digit = X_test[0:1].values
prediction = classifier.predict(test_digit)

#  cross_validation
def boolToInt(val):
    if val == False:
        return 0
    else:
        return 1
y_train5 = y_train5.apply(boolToInt)
scores = cross_val_score(classifier, X_train, y_train5, scoring='accuracy', cv=10, n_jobs=-1)

#  print(scores.mean()) # 0.962

from sklearn.base import BaseEstimator

class Never5Estimator(BaseEstimator):
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never5 = Never5Estimator()

scores_never5 = cross_val_score(never5, X_train, y_train5, scoring='accuracy', cv=10, n_jobs=-1)
#  print(scores_never5.mean()) #  0.90965

#  CUSTOM CROSS VALIDATION IMPLEMENTATION
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

# precisions = []
for train_idx, test_idx in skfolds.split(X_train, y_train5):
    print(f'train idx: {train_idx}, test_idx: {test_idx}')
    clone_clf = clone(classifier)
    X_train_fold = X_train.iloc[train_idx, :]
    X_test_fold = X_train.iloc[test_idx, :]

    y_train5_fold = y_train5[train_idx]
    y_test5_fold = y_train5[test_idx]

    clone_clf.fit(X_train_fold, y_train5_fold)
    y_pred = clone_clf.predict(X_test_fold)
    #
    # precision = metrics.precision_score(y_test5_fold, y_pred)
    # precisions.append(precision)
  # n_correct = sum(y_pred == y_test5_fold)
  # print(n_correct / len(y_pred))

#  MODEL EVALUATION - PRECISION, RECALL CONFUSION MATRIX, GRAPHS
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
#[[TN   FP]
# [FN   TP]]

y_train_preds = cross_val_predict(classifier, X_train, y_train5, cv=3) #  Returns cross validation predictions (not scores)
conf_matrix = confusion_matrix(y_train5, y_train_preds)

precision = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1])
recall = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1])

f1_score = metrics.f1_score(y_train5, y_train_preds)

#  Precision-Recall score tradeoff

y_scores = cross_val_predict(classifier, X_train, y_train5, cv=3, method = "decision_function") #  returns decision scores from SGDClassifier
all_precisions, all_recalls, thresholds = metrics.precision_recall_curve(y_train5, y_scores)

import matplotlib.pyplot as plt

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label='precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label='Recall')
    plt.axis([-20000,20000,0,1])
    plt.legend()
    plt.xlabel('Threshold')
    plt.grid()
    plt.show()

def plot_precision_vs_recall(precisions,recalls):
    plt.plot(recalls[:-1], precisions[:-1], 'b--')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.grid()
    plt.show()

#  plot_precision_recall_vs_threshold(all_precisions, all_recalls, thresholds)
#  plot_precision_vs_recall(all_precisions, all_recalls)

#  Minumum 90% precision
threshold_90_precision = thresholds[np.argmax(np.array(all_precisions) >= 0.90)]
y_train_pred_90 = (y_scores >= threshold_90_precision)

precision_y_train_pred_90 = metrics.precision_score(y_train5, y_train_pred_90)
recall_y_train_pred_90 = metrics.recall_score(y_train5, y_train_pred_90)

print(precision_y_train_pred_90) # 90% precision
print(recall_y_train_pred_90) # 30% recall (very low)

#  ROC CURVE
fpr, tpr, thresholds = metrics.roc_curve(y_train5, y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth = 2, label = label)
    plt.plot([0,1], [0,1], 'k--')

    plt.ylabel('True Positive Rate (TPR)')
    plt.xlabel('False Positive Rate (FPR)')

plot_roc_curve(fpr, tpr)
auc_sgd = metrics.roc_auc_score(y_train5, y_scores) #  area under curve

#  Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state = 42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train5, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]

fpr_forest, tpr_forest, thresholds_forest = metrics.roc_curve(y_train5, y_scores_forest)

plot_roc_curve(fpr, tpr, label = 'SGD')
plot_roc_curve(fpr_forest, tpr_forest, label = 'Random Forest')
plt.legend(loc = 'lower right')
plt.grid()
plt.show()

auc_random_forest = metrics.roc_auc_score(y_train5, y_scores_forest)

y_pred_forest = cross_val_predict(forest_clf, X_train, y_train5, cv=3)
conf_matrix_randomforest = metrics.confusion_matrix(y_train5, y_pred_forest)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  ovr = OneVsRestClassifier(RandomForestClassifier(random_state=55), n_jobs=-1)
#  y_train_pred_ovr = cross_val_predict(ovr, X_train, y_train, cv=5)
#  conf_matrix_ovr = metrics.confusion_matrix(y_train, y_train_pred_ovr)
# plt.matshow(conf_matrix_ovr, cmap=plt.cm.grey)
# plt.show()































































