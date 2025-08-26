import pandas as pd


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
