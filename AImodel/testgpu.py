import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix
import itertools


# confusion matrix 그리는 함수
def plot_confusion_matrix(con_mat, labels, title='Confusion Matrix',
cmap=plt.cm.get_cmap('Blues'), normalize=False):
    plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    marks = np.arange(len(labels))
    nlabels = []
    for k in range(len(con_mat)):
        n = sum(con_mat[k])
        # nlabel = '{0}(n={1})'.format(labels[k],n)
        nlabel = '{0}'.format(labels[k])
        nlabels.append(nlabel)
    plt.xticks(marks, labels)
    plt.yticks(marks, nlabels)
    thresh = con_mat.max() / 2.
    if normalize:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, '{0}%'.format(con_mat[i, j] * 100 / n), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] >thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

fashion_mnist = keras.datasets.fashion_mnist
#from keras.datasets import fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))



train_images = train_images / 255.0
test_images = test_images / 255.0

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=2) #epoch 5


test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_loss=', test_loss)
print('test_acc=',test_acc)


predictions=model.predict(test_images)
pred_labels = np.argmax(predictions, axis=1)
labels =["T Shirt","Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
"Sneaker", "Bag", "Ankle boot"]
true_labels = test_labels.tolist()
#confusion_matrix = confusion_matrix(true_labels, pred_labels, normalize='true')
confusion_matrix = confusion_matrix(true_labels, pred_labels)
print(confusion_matrix)
#confusion_matrix = confusion_matrix(true_labels, pred_labels)
plot_confusion_matrix(confusion_matrix, labels=labels, normalize=True)
