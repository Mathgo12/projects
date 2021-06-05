import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml as fo
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# DATA CREATION AND PREPROCESSING
mnist = fo('mnist_784', version=1)
X,y = mnist['data'], mnist['target']
y = y.astype('uint8')
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# BUILD/TRAIN CLASSIFIER
full_classifier = SGD(random_state=87)
y_pred_full = cross_val_predict(full_classifier, X_train, y_train, cv=5, n_jobs=-1)

full_classifier.fit(X_train, y_train)
#  ERROR ANALYSIS

complete_conf_matrix = metrics.confusion_matrix(y_train, y_pred_full)
scaled_conf_matrix = complete_conf_matrix / complete_conf_matrix.sum(axis=1, keepdims=True)
np.fill_diagonal(scaled_conf_matrix, 0)
plt.matshow(scaled_conf_matrix, cmap = plt.cm.gray)
plt.show()

num_indices = (y_train == 5)
num_preds = y_pred_full[num_indices]
nums = X_train[num_indices]

wrong_idxs = []
for i,j in enumerate(num_preds):
    if j!=5:
        wrong_idxs.append(i)

# indices: 0 to 5851
j = wrong_idxs[5]
plt.matshow(nums[j].reshape(28,28), cmap=plt.cm.gray)
plt.title(f'Predicted as {num_preds[j]}')
plt.show()

precision = metrics.precision_score(y_train, y_pred_full, average='macro')
recall = metrics.recall_score(y_train, y_pred_full, average='macro')
f_score = metrics.f1_score(y_train, y_pred_full, average='macro')

decision_scores = full_classifier.decision_function(X_train[y_train == 0])[:,0]
def toBinary(e):
    if e==0:
        return True
    else:
        return False

fpr, tpr, thresholds = metrics.roc_curve(y_train[(y_train==0)].apply(toBinary), decision_scores)

def plot_roc_curve(fpr, tpr, label=None, auc=None):
    plt.plot(fpr, tpr, "b--",label=label,)
    plt.plot([0,1],[0,1], 'k--')
    plt.ylabel('True Positive Rate (TPR)')
    plt.xlabel('False Positive Rate (FPR)')

plot_roc_curve(fpr, tpr, label="SGDClassifier ROC Curve")
plt.show()

print(metrics.roc_auc_score(y_train, decision_scores))

