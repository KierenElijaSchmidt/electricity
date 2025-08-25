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

    def load_data(self):
        """
        Load dataset and apply initial preprocessing steps.
        """
        df = pd.read_csv(self.filepath)

        # Parse dates and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Map binary columns
        df['holiday'] = df['holiday'].map({'Y': 1, 'N': 0})
        df['school_day'] = df['school_day'].map({'Y': 1, 'N': 0})

        self.df = df
        return self.df

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

    def train_model(self, test_size: float = 0.3, random_state: int = 42):
        """
        Train a LinearRegression model with preprocessing and evaluation.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if self.preproc is None:
            self.build_preprocessor()

        X = self.df.drop(columns=["RRP"])
        y = self.df["RRP"]

        X_preproc = self.preproc.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_preproc, y, test_size=test_size, random_state=random_state
        )

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        test_score = self.model.score(X_test, y_test)
        cv_results = cross_validate(self.model, X_preproc, y, cv=5, scoring="r2")

        return {
            "model": self.model,
            "test_score": test_score,
            "cv_scores": cv_results["test_score"],
            "cv_mean": cv_results["test_score"].mean(),
        }
