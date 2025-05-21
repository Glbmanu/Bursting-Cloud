import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from datetime import date, timedelta

# Set a random seed for reproducibility
np.random.seed(42)

# Generate synthetic features
n_samples = 1000  # Number of data points
n_features = 5    # Number of weather-related features

# Generate random dates for the dataset
start_date = date(2023, 1, 1)
end_date = date(2023, 12, 31)
date_list = [start_date + timedelta(days=np.random.randint((end_date - start_date).days)) for _ in range(n_samples)]

# Generate synthetic cloud burst labels (0: No Cloud Burst, 1: Cloud Burst)
y = np.random.randint(2, size=n_samples)

# Generate synthetic weather-related features
X, _ = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)

# Create a DataFrame for the synthetic dataset
columns = ['Date', 'Temperature', 'Humidity', 'WindSpeed', 'Pressure', 'Precipitation']
df = pd.DataFrame(X, columns=columns[1:])  # Exclude 'Date' column for now
df['Date'] = date_list
df['CloudBurst'] = y

# Save the synthetic dataset to a CSV file
df.to_csv('synthetic_weather_dataset.csv', index=False)

