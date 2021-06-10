import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os
from sklearn import preprocessing
from sklearn import model_selection as ms

basepath = os.getcwd()
folderpath = r"DATA\dogs-vs-cats"
folderpath = os.path.join(basepath, folderpath)
trainfolder = folderpath + r'\train\train'

def getData(datafolder, shape=(200,200)):
    data = []
    filenames = []

    for filename in os.listdir(datafolder):
        img = cv2.imread(os.path.join(datafolder,filename), cv2.IMREAD_COLOR)
        img = cv2.resize(img, shape, interpolation=cv2.INTER_LINEAR)
        data.append(img)
        filenames.append(filename)

    data = np.asarray(data)
    return [data, filenames]

def getLabels(filenames, category1, category2):
    labels = []

    for fn in filenames:
        if str(category1) in fn:
            labels.append(1.0)
        elif str(category2) in fn:
            labels.append(0)

    labels = np.asarray(labels)
    return labels

def preprocess(input_, labels, test_size = 0.1, random_state=45):
    X_train, X_test, y_train, y_test = ms.train_test_split(input_, labels, test_size=test_size, random_state=random_state)

    scaler = preprocessing.StandardScaler()
    for i in range(3):
        X_train[:,:,:,i] = scaler.fit_transform(X_train[:,:,:,i].reshape(len(X_train), 40000)).reshape(len(X_train),200,200)
        X_test[:,:,:,i] = scaler.transform(X_test[:,:,:,i].reshape(len(X_test), 40000)).reshape(len(X_test),200,200)

    return [X_train, X_test, y_train, y_test]

categ_1 = "cat"
categ_2 = "dog"
data, filenames = getData(trainfolder)
labels = getLabels(filenames, categ_1, categ_2)

test_size = 0.1
random_state = 45
shape = (200,200)

X_train, X_test, y_train, y_test = preprocess(data, labels)



