import numpy as np
import pandas as pd
import scipy.stats as sps

distribution_is_normal = lambda data: sps.kstest(data, "norm").pvalue >= 0.05
has_outliers = lambda data: get_outliers(data).any()

def get_outliers(data):
    if distribution_is_normal(data):
        return np.abs(sps.zscore(data)) > 3
    else:
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        return (data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)