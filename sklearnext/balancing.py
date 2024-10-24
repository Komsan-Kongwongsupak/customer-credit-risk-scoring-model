# Import necessary libraries.
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# Define a custom class for balancing binary classification data by either oversampling or downsampling.
class BinaryDataBalancer(BaseEstimator, TransformerMixin):
    # Initialize the balancer with the target column name, the desired ratio, and optional positive/negative class sizes.
    def __init__(self, target_name, ratio, positive=None, negative=None):
        # Ensure that the ratio is a numeric value.
        try:
            assert type(ratio) in [int, float] and ratio is not None
            self.ratio = ratio
        except AssertionError:
            raise TypeError("ratio must be a numeric value.")
        
        # Ensure only one of positive/negative is set (or neither), but not both.
        try:
            assert any([arg is None for arg in [positive, negative]])
            self.positive = positive
            self.negative = negative
        except AssertionError:
            raise TypeError("If majority has been set, minority doesn't need to be set.")
        
        # Ensure the target column name is a string.
        try:
            assert type(target_name) == str
            self.target_name = target_name
        except AssertionError:
            raise TypeError("target_name value must be string.")
    
    # The fit method determines the appropriate positive/negative sample sizes based on the desired ratio.
    def fit(self, X, y=None):
        # Copy the input DataFrame and count the number of positive and negative examples.
        features = X.copy()
        positives = len(features[features[self.target_name] == 1].index)
        negatives = len(features[features[self.target_name] == 0].index)

        # If the positive class size is not provided, calculate it based on the negative class size and the ratio.
        if self.positive is None:
            if self.negative is None:
                # Try both positive and negative target values to maintain the desired ratio.
                positive_1 = positives
                negative_1 = positive_1 / self.ratio
                negative_2 = negatives
                positive_2 = self.ratio * negative_2

                # Choose the sample sizes that result in the smallest difference between the two.
                if abs(positive_1 - positive_2) < abs(negative_1 - negative_2):
                    self.positive = positive_2
                    self.negative = negative_2
                elif abs(positive_1 - positive_2) > abs(negative_1 - negative_2):
                    self.positive = positive_1
                    self.negative = negative_1
                else:
                    # If the differences are equal, set them as the average of both options.
                    self.positive = (positive_1 + positive_2) / 2
                    self.negative = self.positive * self.ratio
            else:
                # If the negative class size is provided, calculate the positive size using the ratio.
                self.positive = self.ratio * self.negative
        else:
            # If the positive size is provided, calculate the negative size using the ratio.
            self.negative = self.positive / self.ratio
        
        # Convert positive and negative sizes to integers.
        self.positive = int(self.positive)
        self.negative = int(self.negative)
        return self

    # The transform method balances the dataset using SMOTE (for oversampling) or resampling (for downsampling).
    def transform(self, X, y=None):
        # Copy the input DataFrame, separate features and target.
        features = X.copy()
        X = features.drop(self.target_name, axis=1)
        y = features[self.target_name]
        
        # Separate positive and negative examples.
        df_positive = features[features[self.target_name] == 1]
        df_negative = features[features[self.target_name] == 0]
        
        # Handle oversampling or downsampling for positive class if necessary.
        if len(df_positive) < self.positive:
            # Use SMOTE to oversample the positive class to the desired size.
            smote = SMOTE(sampling_strategy={1: self.positive}, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        elif len(df_positive) > self.positive:
            # Downsample the positive class to the desired size.
            df_positive_downsampled = resample(
                df_positive,
                replace=False,
                n_samples=self.positive,
                random_state=42
            )
            # Concatenate the downsampled positive class with the negative class.
            df = pd.concat([df_positive_downsampled, df_negative])
            X_resampled = df.drop(columns=self.target_name)
            y_resampled = df[self.target_name]
        else:
            # If no resampling is needed, use the original data.
            X_resampled, y_resampled = X, y
        
        # Combine resampled data into a DataFrame for easier manipulation.
        df = pd.concat([X_resampled, y_resampled], axis=1)
        df_positive = df[df[self.target_name] == 1]
        df_negative = df[df[self.target_name] == 0]
        
        # Handle oversampling or downsampling for negative class if necessary.
        if len(df_negative) < self.negative:
            # Use SMOTE to oversample the negative class to the desired size.
            smote = SMOTE(sampling_strategy={0: self.negative}, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_resampled, y_resampled)
        elif len(df_negative) > self.negative:
            # Downsample the negative class to the desired size.
            df_negative_downsampled = resample(
                df_negative,
                replace=False,
                n_samples=self.negative,
                random_state=42
            )
            # Concatenate the downsampled negative class with the positive class.
            df_resampled = pd.concat([df_positive, df_negative_downsampled])
            X_resampled = df_resampled.drop(columns=self.target_name)
            y_resampled = df_resampled[self.target_name]
        
        # Return the resampled features (X) and target (y).
        return X_resampled, y_resampled
