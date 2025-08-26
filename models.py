from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression




def linear_model(self, test_size: float = 0.3, random_state: int = 42):
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
