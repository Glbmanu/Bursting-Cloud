import numpy as np

# Define the number of samples and features
n_samples = 1000  # Number of data points
n_features = 6    # Number of weather-related features

# Generate synthetic features for cloud burst and non-cloud burst data
# Replace this with your actual data generation logic
def generate_synthetic_data(n_samples, n_features):
    data = np.random.rand(n_samples, n_features)  # Replace with your data generation logic
    return data

# Generate synthetic cloud burst data
cloud_burst_data = generate_synthetic_data(n_samples, n_features)

# Generate synthetic non-cloud burst data
non_cloud_burst_data = generate_synthetic_data(n_samples, n_features)

# Save the synthetic datasets to files
np.save('cloud_burst_training_data.npy', cloud_burst_data)
np.save('non_cloud_burst_training_data.npy', non_cloud_burst_data)

# Generate synthetic test data
n_test_samples = 100  # Number of test data points
test_data = generate_synthetic_data(n_test_samples, n_features)

# Save the synthetic test dataset to a file
np.save('test_data.npy', test_data)

