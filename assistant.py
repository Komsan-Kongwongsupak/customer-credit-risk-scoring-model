# Import necessary libraries for numerical operations and statistical tests.
import numpy as np
import pandas as pd
import scipy.stats as sps

# Lambda function to check if data follows a normal distribution using the Kolmogorov-Smirnov test.
distribution_is_normal = lambda data: sps.kstest(data, "norm").pvalue >= 0.05

# Lambda function to check for outliers in the data.
has_outliers = lambda data: get_outliers(data).any()

# Function to identify outliers in the given data.
def get_outliers(data):
    # Check if the data is normally distributed.
    if distribution_is_normal(data):
        # Return True for values with z-scores greater than 3 (outliers).
        return np.abs(sps.zscore(data)) > 3
    else:
        # Calculate the interquartile range (IQR) and return True for outliers based on IQR method.
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        return (data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)
