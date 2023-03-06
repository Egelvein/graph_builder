import keras
from keras import layers
from keras import models
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(0, 100, 0.1)
Y = np.sin(2*np.pi*X/50)*np.sin(2*np.pi*X/73)
plt.plot(X, Y)

model = models.Sequential()
model.add(layers.Dense(64, input_shape=([1,]), activation='tanh'))
model.add(layers.Dense(32, activation='tanh'))
model.add(layers.Dense(1))

model.compile(loss='mse', 
              optimizer='adam', 
              metrics=['acc'])
history = model.fit(X, Y, 
                    epochs = 1000, 
                    batch_size = 50)
model.save('Graph.h5')

data = model.predict(X)
plt.plot(X, data)
plt.title('result')
plt.show()

acc = history.history['acc']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'b', label='Loss')
plt.title('loss')
plt.legend()

plt.show()
