import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression


class Preprocessor:
    def __init__(self, filepath: str):
        """
        Initialize Preprocessor with dataset path.
        """
        self.filepath = filepath
        self.df = None
        self.preproc = None
        self.model = None



    def build_preprocessor(self):
        """
        Build column transformer for numeric and categorical features.
        """
        num_transformer = make_pipeline(
            SimpleImputer(strategy='median'),
            RobustScaler()
        )

        cat_transformer = OneHotEncoder(handle_unknown="ignore")

        num_col = make_column_selector(dtype_include=['float64', 'int64'])
        cat_col = make_column_selector(dtype_include='object')

        self.preproc = make_column_transformer(
            (num_transformer, num_col),
            (cat_transformer, cat_col)
        )
        return self.preproc


