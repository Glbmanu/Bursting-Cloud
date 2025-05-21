import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.callbacks import ModelCheckpoint  # Import the ModelCheckpoint callback
from tensorflow.keras.regularizers import l2

# ... (previous code)

# Generate a synthetic dataset with weather-related features and cloud burst labels
np.random.seed(42)
n_samples = 1000  # Number of data points
n_features = 6    # Number of weather-related features

# Generate synthetic features (replace this with your data)
X = np.random.rand(n_samples, n_features)

# Generate synthetic cloud burst labels (0: No Cloud Burst, 1: Cloud Burst)
y = np.random.randint(2, size=n_samples)

# Shuffle the data
shuffle_index = np.random.permutation(X.shape[0])
X = X[shuffle_index]
y = y[shuffle_index]

# Define a list of optimizers, kernel sizes, pooling sizes, and learning rates
optimizers = ['sgd']
kernel_sizes = [(1, 1), (2, 2), (3, 3)]
pooling_sizes = [(1, 1), (2, 2), (3, 3)]
learning_rates = [0.0001, 0.0005, 0.001]

# Define a directory to save the best models
model_save_dir = 'saved_models/'

for opt in optimizers:
    for k_size in kernel_sizes:
        for p_size in pooling_sizes:
            for lr in learning_rates:
                try:
                    # Create a new model for each hyperparameter combination
                    model = Sequential()
                    model.add(InputLayer(input_shape=(n_features,)))
                    model.add(Dense(128, activation='relu'))
                    model.add(Dense(256, activation='relu'))
                    model.add(Dense(512, activation='relu'))
                    model.add(Dense(1, activation='sigmoid'))

                    optimizer = None
                    if opt == 'adam':
                        optimizer = Adam(learning_rate=lr)
                    elif opt == 'sgd':
                        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

                    # Compile the model with binary cross-entropy loss function
                    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

                    # Define a ModelCheckpoint callback to save the best model based on validation accuracy
                    checkpoint = ModelCheckpoint(
                        model_save_dir + f'best_model_{opt}_{k_size}_{p_size}_{lr}.h5',
                        monitor='val_accuracy',
                        save_best_only=True,
                        mode='max',  # Save when validation accuracy improves
                        verbose=1
                    )

                    # Train the model on the synthetic data with 50 epochs and batch size of 32
                    history = model.fit(
                        X, y, epochs=50, batch_size=32, validation_split=0.3,
                        callbacks=[checkpoint]  # Pass the ModelCheckpoint callback
                    )

                    ## Print the output of each combination
                    #print(f"Optimizer: {opt}, Kernel Size: {k_size}, Pooling Size: {p_size}, Learning Rate: {lr}")
                    #print(f"Training accuracy: {history.history['accuracy'][-1]}, Validation accuracy: {history.history['val_accuracy'][-1]}")
                    #print("-" * 50)

                    # Calculate and print the confusion matrix

                    y_pred = np.round(model.predict(X)).flatten()
                    y_true = y.flatten()
                    cm = confusion_matrix(y_true, y_pred)
                    #print("Confusion Matrix:")
                    #print(cm)

                    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
                    #print("Accuracy:", accuracy)

                    # Generate a synthetic test dataset (replace this with your test data)
                    n_test_samples = 100  # Number of test data points
                    X_test = np.random.rand(n_test_samples, n_features)

                    # Make predictions for the test data
                    test_predictions = model.predict(X_test)

                    # Multiply the output by 100 and convert it to integers
                    prediction_int = (test_predictions * 100).astype(int)

                    ## Print the result
                    #print("Test Predictions:")
                    #print(prediction_int)
                    # ... (rest of the code)

                except Exception as e:
                    # If there's an error, print a message and continue to the next combination
                    print(f"Error for hyperparameter combination optimizer={opt}, kernel_size={k_size}, pooling_size={p_size}, learning_rate={lr}: {e}")
                    continue

