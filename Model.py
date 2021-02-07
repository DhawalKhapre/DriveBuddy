import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense,MaxPooling2D
from keras.callbacks import TensorBoard

NAME = 'Eye_Classifier'

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

pickle_in = open("Pickle/X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("Pickle/y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0
y = np.array(y)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
#32 and 64 convolution filters used each of size 3x3
#again
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

#128 convolution filters used each of size 3x3
#choose the best features via pooling
    
#randomly turn neurons on and off to improve convergence
    Dropout(0.25),
#flatten since too many dimensions, we only want a classification output
    Flatten(),
#fully connected to get all relevant data
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
#one more dropout for convergence' sake
    Dropout(0.5),
#output a sigmoid to squash the matrix into output probabilities
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=50, validation_split=0.1, callbacks=[tensorboard])

model.save('Model/cnnCat2.h5', overwrite=True)
