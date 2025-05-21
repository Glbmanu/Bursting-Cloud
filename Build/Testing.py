import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from keras.models import load_model

# Load the data
cloudburst_data = np.load('cloud_burst_training_data.npy')
non_cloudburst_data = np.load('non_cloud_burst_training_data.npy')
test_data = np.load('test_data.npy')

#print(cloudburst_data.shape)
#print(non_cloudburst_data.shape)

# Combine the cloudburst and non-cloudburst data into one array
X = np.concatenate((cloudburst_data, non_cloudburst_data))

# Create a target vector (1 for cloudburst, 0 for non-cloudburst)
y = np.concatenate((np.ones(cloudburst_data.shape[0]), np.zeros(non_cloudburst_data.shape[0])))

# Shuffle the data
shuffle_index = np.random.permutation(X.shape[0])
X = X[shuffle_index]
y = y[shuffle_index]

#print(X.shape)
#print(y.shape)

# Visualize training data (handles numerical data)
def visualize_numerical_data(data, labels, title):
    # Visualize numerical data using appropriate plots (e.g., histograms)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
    for i in range(6):
        row = i // 3
        col = i % 3
        axs[row, col].hist(data[:, i], bins=20)
        axs[row, col].set_title(labels[i])
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# Visualize training data (numerical format)
labels = ["RAIN FALL CUM. SINCE 0300 UTC (mm)", "TEMP. ('C)", "RH (%)", "WIND SPEED 10 m (Kt)", "SLP (hPa)", "MSLP (hPa / gpm)"]
visualize_numerical_data(cloudburst_data, labels, title="Cloud Burst Training Data")
visualize_numerical_data(non_cloudburst_data, labels, title="Non-Cloud Burst Training Data")

# Visualize test data (numerical format)
visualize_numerical_data(test_data, labels, title="Test Data")

# Load the model
model = load_model('./saved_models/best_model_sgd_(1, 1)_(3, 3)_0.0005.h5')

# Calculate and print the confusion matrix
y_pred = np.round(model.predict(X)).flatten()
y_true = y.flatten()
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
print("Accuracy:", accuracy)

# Calculate and print the F1-score and recall
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)

print(test_data.shape)
prediction = model.predict(test_data)

# Multiply the output by 100 and convert it to integers
prediction_int = (prediction * 100).astype(int)

# Print the result
#print(prediction_int)

