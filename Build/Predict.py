import streamlit as st
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
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.optimizers import Adam
from keras.models import Sequential

# Load the data
cloudburst_data = np.load('cloud_burst_training_data.npy')
non_cloudburst_data = np.load('non_cloud_burst_training_data.npy')

# Combine the cloudburst and non-cloudburst data into one array
X = np.concatenate((cloudburst_data, non_cloudburst_data))

# Create a target vector (1 for cloudburst, 0 for non-cloudburst)
y = np.concatenate((np.ones(cloudburst_data.shape[0]), np.zeros(non_cloudburst_data.shape[0])))

# Shuffle the data
shuffle_index = np.random.permutation(X.shape[0])
X = X[shuffle_index]
y = y[shuffle_index]

# Load the model
model = load_model('./saved_models/best_model_sgd_(1, 1)_(3, 3)_0.0005.h5')


def generate_GAF(data):
    feature_names = ["RAIN FALL CUM. SINCE 0300 UTC (mm)","TEMP. ('C)","RH (%)","WIND SPEED 10 m (Kt)","SLP (hPa)","MSLP (hPa / gpm)"]

    gaf_images = []
    for feature_name in feature_names:
        # Get the data for the current feature
        data_feature = data[0][feature_name]
        # Fill NaN values with the mean of the column
        data_feature.fillna(data_feature.mean(), inplace=True)
        # Convert the data to a numpy array
        data_feature = data_feature.values
        # Create a Gramian Angular Field object
        gaf = GramianAngularField(image_size=256, method='summation')
        # Convert the data to a GAF image
        image = gaf.transform([data_feature])
        gaf_images.append(image[0])
    return gaf_images


current_date = datetime.date.today() #Current Date
prev_date = current_date - datetime.timedelta(days=6) #Date 6 Days before
# Get the Location Information
state = "UTTARAKHAND"
district = "DEHRADUN"
station = "MUSSOORIE(UKG)_UKG"
#Get the Data from IMD
url = "http://aws.imd.gov.in:8091/AWS/dataview.php?a=AWS&b={}&c={}&d={}&e={}&f={}&g=ALL_HOUR&h=ALL_MINUTE".format(state,district,station,prev_date, current_date)
df_list = pd.read_html(url)

# Take a single test sample from your synthetic dataset (replace with your test data)
# For example, let's use the first test sample
test_sample = X[0].reshape(1, -1)

# Make predictions on the test sample
prediction = model.predict(test_sample)

# Multiply the output by 100 to get the probability as a percentage
probability_percentage = (prediction * 100)[0][0]

# Print the result as a formatted string
print("\nState =", state)
print("District =", district)
print("Station =", station)
print(f"Probability of cloud burst: {probability_percentage:.2f}%")
