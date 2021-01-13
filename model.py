
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, AveragePooling1D, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Sequential

import numpy as np
from data import build_dataset, build_evaluation_dataset, sample, PH_LIMIT
import matplotlib.pyplot as plt



###
# Models
###

def build_model(nb_values):

    model = Sequential()

    model.add(BatchNormalization(input_shape=(nb_values, 1)))
    model.add(Conv1D(filters=32, kernel_size=5, strides=1, input_shape=(nb_values, 1)))
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


def build_model_2(nb_values):

    model = Sequential()

    model.add(BatchNormalization(input_shape=(nb_values, 1)))
    model.add(Conv1D(filters=64, kernel_size=5, padding='same'))
    model.add(AveragePooling1D(pool_size=5))

    model.add(BatchNormalization(input_shape=(nb_values, 1)))
    model.add(Conv1D(filters=128, kernel_size=5, padding='same'))
    model.add(AveragePooling1D(pool_size=5))

    model.add(BatchNormalization())
    model.add(Conv1D(filters=128, kernel_size=5, padding='same', input_shape=(nb_values, 1)))
    model.add(AveragePooling1D(pool_size=8))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model



###
# Model estimation and evaluation
###

nb_values_total, nb_values_part = 4 * 60 * 90, 4 * 60 * 20
nb_samples = 10000

# Build dataset
dataset = build_dataset(nb_values_total, nb_values_part)
d_train_0, d_train_1, d_test_0, d_test_1 = sample(dataset, nb_samples, nb_values_part)

X_train = np.concatenate((d_train_0[0], d_train_1[0]))
X_test = np.concatenate((d_test_0[0], d_test_1[0]))
Y_train = np.concatenate((d_train_0[1], d_train_1[1]))
Y_test = np.concatenate((d_test_0[1], d_test_1[1]))

# Model estimation
model = build_model_2(nb_values_part)
model.fit(X_train, Y_train, batch_size=16, epochs=50, validation_data=(X_test, Y_test), shuffle=True)

# Evaluation
df_train, df_test = build_evaluation_dataset(nb_values_total, nb_values_part)

x_train, y_train = np.expand_dims(np.stack(df_train.fhr.values), 2), np.where(df_train.ph>=PH_LIMIT, 0, 1)
x_test, y_test= np.expand_dims(np.stack(df_test.fhr.values), 2), np.where(df_test.ph>=PH_LIMIT, 0, 1)
y_pred_train = model.predict(x_train).reshape(-1)
y_pred_test = model.predict(x_test).reshape(-1)

print((np.where(y_pred_train>=0.5, 1, 0)==y_train).sum() / len(y_train))
print((np.where(y_pred_test>=0.5, 1, 0)==y_test).sum() / len(y_test))


##############

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



