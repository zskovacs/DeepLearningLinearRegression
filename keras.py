# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import os
from pathlib import Path

model1 = { 'optimizer': 'sgd', 'loss': 'mse' } # 75.39%
model2 = { 'optimizer': 'adam', 'loss': 'binary_crossentropy' } #83.85%
model3 = { 'optimizer': 'sgd', 'loss': 'binary_crossentropy' } #76.30%
model4 = { 'optimizer': 'adam', 'loss': 'mse' } #85.55%
model5 = { 'optimizer': 'RMSprop', 'loss': 'mse' } #81.51% #For Recurrent networks

selectedModel = model4
selectedModel['fileName'] = 'models/model_' + selectedModel['optimizer'] + '_' + selectedModel['loss']

modelfile = Path(selectedModel['fileName'] + '.json')
weightsfile = Path(selectedModel['fileName'] + '.h5')

# fix random seed for reproducibility
np.random.seed(7)
# load pima indians dataset
dataset = np.loadtxt("keras.ds.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

if modelfile.is_file() and weightsfile.is_file():
    # load json and create model
    json_file = open(modelfile.name, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights(weightsfile.name)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    model.compile(loss=selectedModel['loss'], optimizer=selectedModel['optimizer'], metrics=['accuracy'])
    score = model.evaluate(X, Y, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
else:
    # create model
    model = Sequential()
    model.add(Dense(100, input_dim=8, kernel_initializer='uniform', activation='relu', use_bias=True))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu', use_bias=True))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss=selectedModel['loss'], optimizer=selectedModel['optimizer'], metrics=['accuracy'])
    # Fit the model
    model.fit(X, Y, epochs=150, batch_size=10, verbose=1)
    # evaluate the model
    scores = model.evaluate(X, Y, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # serialize model to JSON
    model_json = model.to_json()
    with open(modelfile.name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weightsfile.name)
    print("Saved model to disk")


test_data = np.asarray([1,85,66,27,0,26.6,0.352,31]).reshape(1,8)
predictions = model.predict(test_data)
print("prediction=",predictions[0])
