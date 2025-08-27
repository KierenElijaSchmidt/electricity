from electricity.preprocessing import Preprocessor 
from electricity.load import Loading 

DATA_FILE = "complete_dataset.csv"  # file lives in repo root alongside load.py

loader = Loading(
    filepath=DATA_FILE,
    create_time_features=False,   # avoid duplicating with preprocessing.DateCyclicalFeatures
    return_X_y=True               # directly return X, y
)
