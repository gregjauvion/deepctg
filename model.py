
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, AveragePooling1D, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
from data import get_record_names, read, process_fhr, NB_VALUES
import matplotlib.pyplot as plt

# FHR
# CU
# Terme (41 semaines par ex.)
# --> Prédire le pH, apgar
# Pour commencer les 15 (30?) dernières minutes, mais pas plus d'une heure
# ph>=7.15


def build_model():

    model = Sequential()

    model.add(BatchNormalization(input_shape=(NB_VALUES, 1)))
    model.add(Conv1D(filters=32, kernel_size=5, strides=1, input_shape=(NB_VALUES, 1)))
    model.add(AveragePooling1D(pool_size=5))

    model.add(BatchNormalization())
    model.add(Conv1D(filters=16, kernel_size=5, strides=1))
    model.add(AveragePooling1D(pool_size=5))

    model.add(BatchNormalization())
    model.add(Conv1D(filters=8, kernel_size=5, strides=1))
    model.add(AveragePooling1D(pool_size=5))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def build_model_2():

    model = Sequential()

    model.add(BatchNormalization(input_shape=(NB_VALUES, 1)))
    model.add(Conv1D(filters=64, kernel_size=5, padding='same'))
    model.add(AveragePooling1D(pool_size=5))

    model.add(BatchNormalization(input_shape=(NB_VALUES, 1)))
    model.add(Conv1D(filters=32, kernel_size=5, padding='same'))
    model.add(AveragePooling1D(pool_size=5))

    model.add(BatchNormalization())
    model.add(Conv1D(filters=16, kernel_size=5, padding='same', input_shape=(NB_VALUES, 1)))
    model.add(AveragePooling1D(pool_size=8))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


# Build dataset
records = get_record_names()
data = [read(r) for r in records]

fhr = [process_fhr(i[0]) for i in data]
hypoxia = np.array([i[2]<7.15 for i in data])

X = np.stack([np.expand_dims(f, 1) for f in fhr if f is not None])
Y = np.array([1 if h else 0 for f, h in zip(fhr, hypoxia) if f is not None])

nb_train = 450
X_train, X_test = X[:nb_train], X[nb_train:]
Y_train, Y_test = Y[:nb_train], Y[nb_train:]

model = build_model_2()
model.fit(X_train, Y_train, batch_size=16, epochs=500, validation_data=(X_test, Y_test))



Y_train_pred = model.predict(X_train).reshape(-1)
indices = np.argsort(Y_train_pred)
plt.scatter(range(len(indices)), Y_train_pred[indices])
plt.scatter(range(len(indices)), Y_train[indices])
plt.show()




Y_test_pred = model.predict(X_test).reshape(-1)
indices = np.argsort(Y_test_pred)
plt.scatter(range(len(indices)), Y_test_pred[indices])
plt.scatter(range(len(indices)), Y_test[indices])
plt.show()



