import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate some example data
np.random.seed(42)
X = np.random.rand(100, 1)  # Create 100 random data points
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)  # y = 2X + 1 + noise

# Create a DataFrame using Pandas
df = pd.DataFrame({'X': X.flatten(), 'y': y.flatten()})

# Visualize the data
plt.scatter(df['X'], df['y'])
plt.xlabel('X')
plt.ylabel('y')
plt.title('Scatter plot of X vs. y')
plt.show()

# Split the data into training and test sets
X_train, X_test, y_train
