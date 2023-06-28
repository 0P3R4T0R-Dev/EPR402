import numpy as np
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(42)

# Generate 300 samples from rand() and randn()
samples_rand = np.random.rand(3000) - 0.5
samples_randn = np.random.randn(3000)

# Create the figure and axis objects
fig, ax = plt.subplots()

# Plot the histogram for samples_rand using 30 bins in blue
ax.hist(samples_rand, bins=30, label='rand()', color='blue', alpha=0.5)

# Plot the histogram for samples_randn using 30 bins in red
ax.hist(samples_randn, bins=30, label='randn()', color='red', alpha=0.5)

# Set the labels and title
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.set_title('Histograms of rand() and randn()')

# Add a legend
ax.legend()

# Display the plot
plt.show()
