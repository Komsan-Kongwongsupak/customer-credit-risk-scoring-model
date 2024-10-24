# Import necessary libraries for numerical operations, data manipulation, and statistical analysis.
import numpy as np
import pandas as pd
import scipy.stats as sps

# Lambda function to check if data follows a normal distribution using the Kolmogorov-Smirnov test.
distribution_is_normal = lambda data: sps.kstest(data, "norm").pvalue >= 0.05

# Lambda function to check if there are any outliers in the data by calling the get_outliers function.
has_outliers = lambda data: get_outliers(data).any()

# Lambda function to check if data is continuous by identifying non-integer and non-categorical values.
is_continuous = lambda data: any([(type(v) not in [int, str, bool]) and (v != int(v)) for v in list(data.unique())])

# Function to identify outliers in the data using z-score for normally distributed data,
# or using the interquartile range (IQR) for non-normal data.
def get_outliers(data):
    # If the data is normally distributed, use z-score to identify outliers (z > 3).
    if distribution_is_normal(data):
        return np.abs(sps.zscore(data)) > 3
    else:
        # Otherwise, use IQR to detect outliers: values beyond 1.5 * IQR from the first and third quartile.
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        return (data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)

# Function to calculate correlation matrix for numerical columns in the DataFrame.
def corr_numerical(df, cols):
    # Use the built-in pandas corr() method to compute Pearson correlation for numerical data.
    return df[cols].corr()

# Function to calculate correlation matrix for categorical columns in the DataFrame using the Chi-square test.
def corr_categorical(df, cols, significant=0.05):
    # Initialize empty lists to store correlation coefficients and p-values.
    data_corr_catg_cov = []
    data_corr_catg_p = []
    
    # Iterate over pairs of categorical columns to calculate their correlation.
    for col1 in cols:
        row_cov = []  # To store the correlation coefficients for col1 with all other columns.
        row_p = []    # To store the p-values for col1 with all other columns.
        for col2 in cols:
            # Create a contingency table for the two categorical variables.
            contingency = pd.crosstab(df[col1], df[col2])
            # Perform the Chi-square test without applying Yates' correction.
            chi2_res = sps.chi2_contingency(contingency, correction=False)
            # Calculate Cramér's V (effect size) from the Chi-square result and add it to row_cov.
            row_cov.append(np.sqrt((chi2_res[0] / np.sum(contingency.values)) / (min(contingency.shape) - 1)))
            # Append the p-value from the Chi-square test to row_p.
            row_p.append(chi2_res[1])

        # Append the correlation and p-value rows to their respective lists.
        data_corr_catg_cov.append(row_cov)
        data_corr_catg_p.append(row_p)
    
    # Convert the lists into DataFrames for easier interpretation: one for Cramér's V, one for p-values.
    df_corr_catg_cov = pd.DataFrame(columns=cols, index=cols, data=data_corr_catg_cov)
    df_corr_catg_p = pd.DataFrame(columns=cols, index=cols, data=data_corr_catg_p)

    # Identify columns where all p-values are above the significance threshold (not significant).
    axis_dropped = []
    for col in df_corr_catg_p.columns:
        if all([e >= significant for e in df_corr_catg_p[col].values]):
            axis_dropped.append(col)  # Mark those columns to drop from the correlation matrix.
    
    # Return the Cramér's V correlation matrix without the non-significant columns.
    return df_corr_catg_cov.drop(columns=axis_dropped, index=axis_dropped)
