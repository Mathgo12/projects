from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV


data = mnist.load_data()
(X_train, y_train), (X_test, y_test) = data
X_train = X_train.reshape(len(X_train), 784)
X_test = X_test.reshape(len(X_test), 784)

k = 3
weights = 'uniform'
kneighbors = KNeighborsClassifier(n_neighbors=k, weights=weights)
kneighbors.fit(X_train, y_train)

y_pred_cv = []
accuracies = []
precisions = []
recalls = []


skf = StratifiedKFold(random_state = 54, n_splits=5, shuffle=True)
for train_idx, test_idx in skf.split(X_train, y_train):
    model = clone(kneighbors)
    X_train_temp = X_train[train_idx, :]
    X_test_temp = X_train[test_idx,:]

    y_train_temp = y_train[train_idx]
    y_test_temp = y_train[test_idx]

    model.fit(X_train_temp, y_train_temp)
    accuracies.append(metrics.accuracy_score(y_test_temp, model.predict(X_test_temp)))


accuracies = np.array(accuracies) # mean is about 0.972

#Randomized Search CV
param_grid = [
    {'n_neighbors': [3,5,10,15], 'weights': ['uniform', 'distance']}
]
n_iter = 5
rs = RandomizedSearchCV(kneighbors, param_grid, n_iter=n_iter, n_jobs=-1, random_state=34)
rs.fit(X_train, y_train)

best_params = rs.best_params_
best_estimator = rs.best_estimator_

#  FINAL STATS
final_prediction = best_estimator.predict(X_test)
final_accuracy = metrics.accuracy_score(y_test, final_prediction)
final_precision = metrics.precision_score(y_test, final_prediction, average='macro')
final_recall = metrics.recall_score(y_test, final_prediction, average='macro')

#  SAVE MODEL
import joblib
joblib.dump(best_estimator, "knn_mnist.pkl")

model = joblib.load('knn_mnist.pkl')
print(model.get_params)

f = open('mydir/SomeText.txt', 'r')
print(f.read())
f.close()
















