import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
from tensorflow.keras.layers import Bidirectional

def train_model():
    X_train = np.load('data/X_train.npy')
    Y_train = np.load('data/Y_train.npy')
    X_test = np.load('data/X_test.npy')
    Y_test = np.load('data/Y_test.npy')

    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)

    # Define LSTM model
    # LSTM is a type of RNN that is capable of learning long-term dependencies
    model = Sequential()
    model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(units=50, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(X_train, Y_train,
              validation_data=(X_test, Y_test),
              epochs=50,
              batch_size=32,
              callbacks=[es])

    if not os.path.exists('model'):
        os.makedirs('model')
    # Using .keras format as suggested by the warnings
    model.save('model/model.keras')
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    train_model()