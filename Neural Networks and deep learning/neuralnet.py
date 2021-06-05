from tensorflow import keras
import sklearn.preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_train_full = X_train_full.reshape(60000,784)
X_test = X_test.reshape(-1,784)

scaler = sklearn.preprocessing.StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

X_valid, X_train = X_train_full_scaled[:5000], X_train_full_scaled[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

#  Subclassing keras.Model
class MyModel(keras.Model):
    def __init__(self, units = 300, activation='relu', **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(10, activation=activation)
        self.mainout = keras.layers.Dense(10, activation='softmax')

    def call(self, input_, training=None, mask=None):   #  Constructs model
        input_flattened = keras.layers.Flatten(input_shape=input_.shape)(input_)
        hidden1 = self.hidden1(input_flattened)
        hidden2 = self.hidden2(hidden1)
        mainout = self.mainout(hidden2)
        return mainout

    def get_config(self):
        base_conf = super().get_config()
        return base_conf

#inputs = keras.layers.Input(shape=[784])
model = MyModel()
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
              optimizer=keras.optimizers.SGD(lr=0.02),
              metrics=['accuracy'])
history = model.fit(X_train, y_train,epochs = 20,validation_data=(X_valid, y_valid))

#model.save('Neural Networks and deep learning/mymodel')

#modelNew = keras.models.load_model('Neural Networks and deep learning/mymodel')

def plot_images(image, label):
    plt.matshow(image, cmap=plt.cm.gray)
    plt.title(f'{label}')
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])


class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]
# EVALUATION
evaluation = model.evaluate(X_test, y_test)
y_pred = modelNew.predict(X_test)
i=27
#plot_images(scaler.inverse_transform(X_test[i].reshape(784)).reshape(28,28), class_names[np.argmax(y_pred[i])])
#plt.show()

#  plots
def evaluation_plot(history, epochs):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    fig = plt.gcf()
    fig.set_size_inches(10, 4)
    plt.get_current_fig_manager().window.resizable(False, False)
    plt.axis([0,epochs,0,1])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy and Loss')
    plt.grid()
    plt.legend(loc='lower right')
    plt.gca().set_ylim(0,1)

evaluation_plot(history, epochs=20)
plt.show()
















































































