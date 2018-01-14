"""
ECS171 HW2
Gabriel Yin #999885129
1.  Sequence Name: Accession number for the SWISS-PROT database
2.  mcg: McGeoch's method for signal sequence recognition.
3.  gvh: von Heijne's method for signal sequence recognition.
4.  alm: Score of the ALOM membrane spanning region prediction program.
5.  mit: Score of discriminant analysis of the amino acid content of
   the N-terminal region (20 residues long) of mitochondrial and
         non-mitochondrial proteins.
6.  erl: Presence of "HDEL" substring (thought to act as a signal for
   retention in the endoplasmic reticulum lumen). Binary attribute.
7.  pox: Peroxisomal targeting signal in the C-terminus.
8.  vac: Score of discriminant analysis of the amino acid content of
         vacuolar and extracellular proteins.
9.  nuc: Score of discriminant analysis of nuclear localization signals
   of nuclear and non-nuclear proteins.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras, tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder

"""
Class WeightHistory: Customerized CallBacks Used For Weights History

@ __init__ :
  self: optional
  model: take the current model as an argument
  wt: weight list to keep track of target weights
  logs: general storage. optional

@ on_epoch_end :
  takes default three arguments
  record weight on each epoch
"""

class WeightHistory(keras.callbacks.Callback):

    def __init__(self, model, wt, logs = {}):

        self.weights = wt
        self.model = model


    def on_epoch_end(self, epoch, logs = {}):
        self.weights.append(self.model.layers[2].get_weights())
        return

# Problem 1
seed_num = 7
np.random.seed(seed_num)
Weights = []

dataset = pd.read_csv("yeast.data.txt", sep='\s+', header = None)
dataset[9] = dataset[9].replace(['CYT', 'ME1', 'ME3', 'EXC', 'MIT', 'ERL', 'POX', 'NUC','ME2','VAC'],[0,1,2,3,4,5,6,7,8,9])
dataset = dataset.values
dataset_x = dataset[:,1:9].astype(float)
dataset_y = dataset[:,9]

# sequential model
model = Sequential()
# add three layers and use sigmoid activiation
model.add(Dense(8, input_dim = 8, kernel_initializer='normal', activation = "sigmoid"))
model.add(Dense(3, activation = "sigmoid"))
model.add(Dense(10, activation = "sigmoid"))
# number of epoch is 50
slot = 360
learning_rate = 0.1
decay_rate = learning_rate / slot
# define sgd optimizer using SGD method
sgd = SGD(lr = learning_rate, momentum = 0.8, decay = decay_rate, nesterov = False)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
callback = WeightHistory(model, Weights)
# hist is a history object where we can retrieve the history of training
hist = model.fit(dataset_x, dataset_y, validation_split = 0.3, epochs = slot, batch_size = 4, callbacks=[callback], verbose = 2)

# Draw Graph
w1, w2, w3, b1 = [], [], [], [],
errors_t, errors_e = [],[]

for i in range(0, 360):

    w1.append(callback.weights[i][0][0][0]) # weight 1
    w2.append(callback.weights[i][0][1][0]) # weight 2
    w3.append(callback.weights[i][0][2][0]) # weight 3
    b1.append(callback.weights[i][1][0])



for item in hist.history['acc']:
    errors_t.append(1 - item)

for item in hist.history['val_acc']:
    errors_e.append(1 - item)

# Weights
plt.xlabel('number of epoch')
plt.ylabel('weights')
plt.plot(w1)
plt.plot(w2)
plt.plot(w3)
plt.plot(b1)
plt.legend(['weight1', 'weight2', 'weight3', 'bias'], loc = 'upper left')
plt.show()
# errors
plt.xlabel('number of epoch')
plt.ylabel('erros')
plt.plot(errors_t)
plt.plot(errors_e)
plt.legend(['errors_train', 'errors_test'], loc = 'upper left')
plt.show()

# Problem 2
# re-train model with all samples as training set
# sequential model
model2 = Sequential()
# add three layers and use sigmoid activiation
model2.add(Dense(8, input_dim = 8, kernel_initializer='normal', activation = "sigmoid"))
model2.add(Dense(3, activation = "sigmoid"))
model2.add(Dense(10, activation = "sigmoid"))
# number of epoch is 50
slot2 = 360
learning_rate = 0.1
decay_rate = learning_rate / slot2
# define sgd optimizer using SGD method
sgd = SGD(lr = learning_rate, momentum = 0.8, decay = decay_rate, nesterov = False)
model2.compile(loss = 'sparse_categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
# hist is a history object where we can retrieve the history of training
model2.fit(dataset_x, dataset_y, epochs = slot2, batch_size = 4, verbose = 2)

result_error = 1 - model2.evaluate(dataset_x, dataset_y)[1]

# Problem 3: Please see report

# Problem 4:

model4 = Sequential()
# add layers according to the problem description and use sigmoid activiation
# increase layer from 1 -> 2 -> 3
# increase # of nodes from 3 -> 6 -> 9 -> 12
model4.add(Dense(8, input_dim = 8, kernel_initializer='normal', activation = "sigmoid"))
model4.add(Dense(12, activation = "sigmoid"))
#model4.add(Dense(12, activation = "sigmoid"))
#model4.add(Dense(12, activation = "sigmoid"))
model4.add(Dense(10, activation = "sigmoid"))
# number of epoch is 50
slot2 = 360
learning_rate = 0.1
decay_rate = learning_rate / slot2
# define sgd optimizer using SGD method
sgd = SGD(lr = learning_rate, momentum = 0.8, decay = decay_rate, nesterov = False)
model4.compile(loss = 'sparse_categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
# hist is a history object where we can retrieve the history of training
model4.fit(dataset_x, dataset_y, epochs = slot2, batch_size = 4, verbose = 2)

result_error = 1 - model4.evaluate(dataset_x, dataset_y)[1]

# Problem 5

sample = np.array([[0.49, 0.51, 0.52, 0.23, 0.55, 0.03, 0.52, 0.39]])
result = model4.predict_classes(sample) # returns 7. So NUC
