import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class AgeBinner(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, encoder=OneHotEncoder):
        self.columns = columns
        self.encoder = encoder

    def fit(self, X, y=None):
        return self

    def create_age_bins(self, age):
        age = float(age)
        if age < 3:
            return "baby"
        elif 3 <= age < 11:
            return "child"
        elif 11 <= age < 18:
            return "adolescent"
        elif 18 <= age < 30:
            return "young adult"
        elif 30 <= age < 65:
            return "adult"
        return "elderly"

    def transform(self, X, y=None):
        X["age_range"] = X["age"].apply(self.create_age_bins)
        for column in ['age_range_adolescent', 'age_range_adult', 'age_range_baby', 'age_range_child',
                       'age_range_elderly', 'age_range_young adult']:
            if column != ('age_range_'+ X["age_range"][0]):
                X[column] = 0

        X = X.drop(["age"], axis=1)
        X = self.encoder.fit_transform(X)
        self.X = X



        return X

    def get_feature_names_out(self):
        return self.encoder.get_feature_names_out()


class GlucoseLevelBinner(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, encoder=None):  # default params
        self.columns = columns
        self.encoder = encoder

    def find_glucose_range(self, age, glucose_level):
        age = float(age)
        glucose_level = float(glucose_level)

        if age <= 5:
            if glucose_level <= 100:
                return "too low"
            elif 100 < glucose_level < 180:
                return "in range"
            else:
                return "too high"
        if 6 <= age <= 9:
            if glucose_level <= 80:
                return "too low"
            elif 80 < glucose_level < 140:
                return "in range"
            else:
                return "too high"
        if age >= 10:
            if glucose_level <= 70:
                return "too low"
            elif 70 < glucose_level < 140:
                return "in range"
            else:
                return "too high"

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cols_for_binning = self.columns
        X["glucose_range"] = X.apply(
            lambda x: self.find_glucose_range(x[cols_for_binning[0]], x[cols_for_binning[1]]), axis=1
        )

        for column in ['glucose_range_too low', 'glucose_range_in range', 'glucose_range_too high']:
            if column != ('glucose_range_'+ X["glucose_range"][0]):
                X[column] = 0

        X = X.drop(["age", 'avg_glucose_level'], axis=1)
        X = self.encoder.fit_transform(X)
        self.X = X
        return X

    def get_feature_names_out(self):
        return self.encoder.get_feature_names_out()