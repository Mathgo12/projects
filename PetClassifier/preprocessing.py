import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random

import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection as ms
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Activation,BatchNormalization

basepath = os.getcwd()
folderpath = 'DATA/dogs-vs-cats'
folderpath = os.path.join(basepath, folderpath)
datafolder = folderpath + '/allData'
trainfolder = folderpath + '/train'
validationfolder = folderpath + '/validation'

dogs_path = basepath + '/DATA/dogs-vs-cats/train/dogs'
cats_path = basepath + '/DATA/dogs-vs-cats/train/cats'

if not os.path.exists(basepath + '/DATA/dogs-vs-cats/validation'):
    os.mkdir(basepath + '/DATA/dogs-vs-cats/validation')
if not os.path.exists(basepath + '/DATA/dogs-vs-cats/train/cats'):
    os.mkdir(basepath + '/DATA/dogs-vs-cats/train/cats')
elif not os.path.exists(basepath + '/DATA/dogs-vs-cats/train/dogs'):
    os.mkdir(basepath + '/DATA/dogs-vs-cats/train/dogs')

cat_idx = 0
dog_idx = 0

for filename in os.listdir(datafolder):
    img = cv2.imread(os.path.join(trainfolder, filename), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (200,200), interpolation=cv2.INTER_LINEAR)
    if "cat" in filename:
        cv2.imwrite(cats_path + f'/cat-{cat_idx}.jpg', img)
        cat_idx += 1
    else:
        cv2.imwrite(dogs_path + f'/dog-{dog_idx}.jpg', img)
        dog_idx += 1

cats_imgs = os.listdir(cats_path)
dogs_imgs = os.listdir(dogs_path)

random.shuffle(cats_imgs)
random.shuffle(dogs_imgs)

for filename in cats_imgs[-2500:]:
    os.replace(cats_path + f'/{filename}', folderpath + f'/validation/cats/{filename}')

for filename in dogs_imgs[-2500:]:
    os.replace(dogs_path + f'/{filename}', folderpath + f'/validation/dogs/{filename}')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_datasetiter = train_datagen.flow_from_directory(
    directory=trainfolder,
    target_size = (150,150),
    color_mode = 'rgb',
    class_mode= 'binary',
    batch_size = 32

)

validation_datasetiter = test_datagen.flow_from_directory(
    directory=validationfolder,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')


model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer = 'rmsprop', metrics = [keras.metrics.binary_accuracy])

history = model.fit_generator(
    train_datasetiter,
    steps_per_epoch=469,
    epochs=30,
    validation_data=validation_datasetiter,  #  Alternatively: validation_split = 0.2 (fraction of the training data)
    validation_steps=157
)

#  Evaluation
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid()
plt.axis([0,30,0,1])
plt.show()

model.evaluate()




