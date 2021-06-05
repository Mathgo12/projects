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
from scipy.ndimage.interpolation import shift
import joblib

data = mnist.load_data()
(X_train, y_train), (X_test, y_test) = data
X_train = X_train.reshape(len(X_train), 784)
X_test = X_test.reshape(len(X_test), 784)

X_train_rs = np.zeros((10000,28,28))
X_train_ls = np.zeros((10000,28,28))
X_train_bs = np.zeros((10000,28,28))
X_train_ts = np.zeros((10000,28,28))

y_train_new = np.concatenate([y_train,y_train[0:10000],y_train[0:10000],y_train[0:10000],y_train[0:10000]])

X_train_temp = X_train
for i in range(10000):
    X_train_rs[i] = shift(X_train_temp[i].reshape(28, 28), [0, 2])
    X_train_ls[i] = shift(X_train_temp[i].reshape(28, 28), [0, -2], cval=0.0)
    X_train_bs[i] = shift(X_train_temp[i].reshape(28, 28), [2, 0], cval=0.0)
    X_train_ts[i] = shift(X_train_temp[i].reshape(28, 28), [-2, 0], cval=0.0)



X_train_rs=X_train_rs.reshape(10000,784)
X_train_ls=X_train_ls.reshape(10000,784)
X_train_bs=X_train_bs.reshape(10000,784)
X_train_ts=X_train_ts.reshape(10000,784)

X_train_new = np.concatenate([X_train, X_train_rs, X_train_ls, X_train_bs, X_train_ts], axis=0)


k = 3
weights = 'uniform'
kneighbors = KNeighborsClassifier()
kneighbors.set_params(**{'algorithm': 'auto',
 'leaf_size': 30,
 'metric': 'minkowski',
 'metric_params': None,
 'n_jobs': None,
 'n_neighbors': 5,
 'p': 2,
 'weights': 'distance'})
kneighbors.fit(X_train_new, y_train_new)

final_prediction = kneighbors.predict(X_test)
final_accuracy = metrics.accuracy_score(y_test, final_prediction)























