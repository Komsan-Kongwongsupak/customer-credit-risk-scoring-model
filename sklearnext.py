import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

class BinaryDataBalancer(BaseEstimator, TransformerMixin):
    def __init__(self, target_name, ratio, positive=None, negative=None):
        try:
            assert type(ratio) in [int, float] and ratio is not None
            self.ratio = ratio
        except AssertionError:
            raise TypeError("ratio must be a numeric value.")

        try:
            assert any([arg is None for arg in [positive, negative]])
            self.positive = positive
            self.negative = negative
        except AssertionError:
            raise TypeError("If majority has been set, minority doesn't need to be set.")
        
        try:
            assert type(target_name) == str
            self.target_name = target_name
        except AssertionError:
            raise TypeError("target_name value must be string.")
    
    def fit(self, X, y=None):
        features = X.copy()
        positives = len(features[features[self.target_name] == 1].index)
        negatives = len(features[features[self.target_name] == 0].index)

        if self.positive is None:
            if self.negative is None:
                positive_1 = positives
                negative_1 = positive_1/self.ratio
                negative_2 = negatives
                positive_2 = self.ratio*negative_2

                if abs(positive_1 - positive_2) < abs(negative_1 - negative_2):
                    self.positive = positive_2
                    self.negative = negative_2
                elif abs(positive_1 - positive_2) > abs(negative_1 - negative_2):
                    self.positive = positive_1
                    self.negative = negative_1
                else:
                    self.positive = (positive_1 + positive_2)/2
                    self.negative = self.positive*self.ratio
            else:
                self.positive = self.ratio*self.negative
        else:
            self.negative = self.positive/self.ratio

        self.positive = int(self.positive)
        self.negative = int(self.negative)
        return self

    def transform(self, X, y=None):
        features = X.copy()
        X = features.drop(self.target_name, axis=1)
        y = features[self.target_name]
        df_positive = features[features[self.target_name] == 1]
        df_negative = features[features[self.target_name] == 0]
        
        if len(df_positive) < self.positive:
            smote = SMOTE(sampling_strategy={1: self.positive}, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        elif len(df_positive) > self.positive:
            df_positive_downsampled = resample(
                df_positive,
                replace=False,
                n_samples=self.positive,
                random_state=42
            )
            df = pd.concat([df_positive_downsampled, df_negative])
            X_resampled = df.drop(columns=self.target_name)
            y_resampled = df[self.target_name]
        else:
            X_resampled, y_resampled = X, y
        
        df = pd.concat([X_resampled, y_resampled], axis=1)
        df_positive = df[df[self.target_name] == 1]
        df_negative = df[df[self.target_name] == 0]
        
        if len(df_negative) < self.negative:
            smote = SMOTE(sampling_strategy={0: self.negative}, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_resampled, y_resampled)
        elif len(df_negative) > self.negative:
            df_negative_downsampled = resample(
                df_negative,
                replace=False,
                n_samples=self.negative,
                random_state=42
            )
            df_resampled = pd.concat([df_positive, df_negative_downsampled])
            X_resampled = df_resampled.drop(columns=self.target_name)
            y_resampled = df_resampled[self.target_name]
        
        return X_resampled, y_resampled
