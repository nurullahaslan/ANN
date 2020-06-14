import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('data_banknote_authentication.csv')

X= veriler.iloc[:,0:4].values
Y = veriler.iloc[:,4].values


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adagrad

classifier = Sequential()
classifier.add(Dense(16, init = 'uniform', activation = 'relu' , input_dim = 4))#AUC=9.8
classifier.add(Dense(1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(Adagrad(lr=0.01), loss =  'binary_crossentropy' , metrics = ['accuracy'] )
classifier.fit(x_train, y_train,validation_data=(x_test,y_test), epochs=50)
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
cm = confusion_matrix(y_test,y_pred)
class_names = ['FAKE', 'REAL']

fig, ax = plot_confusion_matrix(cm,class_names=class_names)
plt.show()

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

probs = classifier.predict_proba(x_test)
probs = y_pred[:, 0]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr)