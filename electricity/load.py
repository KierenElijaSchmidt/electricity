import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Iterable

@dataclass
class Loading:
    filepath: str | Path
    date_col: str = "date"
    target_col: str = "RRP"
    # columns whose presence would leak label info, to be dropped if found
    leaky_cols: Iterable[str] = (
        "RRP_positive", "RRP_negative",
        "demand_pos_RRP", "demand_neg_RRP",
        "frac_at_neg_RRP",
    )
    bool_maps: dict = field(default_factory=lambda: {
        "holiday": {"Y": 1, "N": 0},
        "school_day": {"Y": 1, "N": 0},
    })
    drop_duplicate_index: bool = True
    create_time_features: bool = True       # index-derived features (safe)
    create_target_lags: bool = False        # off by default; uses target shift() only
    lags: Iterable[int] = (1, 7)            # safe past-only lags
    rolling_windows: Iterable[int] = (7,)   # past-only rolling means of target
    fillna_numeric: Optional[float] = None  # e.g., 0.0; if None, keep NaN
    return_X_y: bool = False                # if True, returns (X, y)

    def load_data(self) -> pd.DataFrame:
        raw_path = Path(self.filepath)
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            raw_path,
            repo_root / "raw_data" / raw_path,
            repo_root / raw_path,
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            tried = ", ".join(str(p) for p in candidates)
            raise FileNotFoundError(f"File not found. Tried: {tried}")

        # Parse date on read
        df = pd.read_csv(path, parse_dates=[self.date_col])

        if self.date_col not in df.columns:
            raise KeyError(f"Required column '{self.date_col}' not found.")

        # Indexing & ordering
        df = df.sort_values(self.date_col).set_index(self.date_col)

        # Map binary columns safely
        for col, mapping in self.bool_maps.items():
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype("string")
                    .str.strip()
                    .str.upper()
                    .map(mapping)            # unexpected -> <NA>
                    .astype("Int8")
                )

        # Drop leaky columns if present
        to_drop = [c for c in self.leaky_cols if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)

        # Optional duplicate index handling
        if self.drop_duplicate_index and not df.index.is_unique:
            df = df[~df.index.duplicated(keep="first")]

        # --- Autoregressive feature: previous day's y value ---
        # Create a new column with the previous value of the target column (lag 1)
        if self.target_col in df.columns:
            df[f"{self.target_col}_t_minus_1"] = df[self.target_col].shift(1)
            
            # Shift all other X columns (except t-1, school_day, holiday, date, and the target itself) by 1,
            # and create new columns named as <col>_t_minus_1
            exclude_cols = {
                f"{self.target_col}_t_minus_1",
                "school_day",
                "holiday",
                self.date_col,
                self.target_col,  # Exclude the target column itself from being dropped
            }
            x_cols = [col for col in df.columns if col not in exclude_cols]
            for col in x_cols:
                df[f"{col}_t_minus_1"] = df[col].shift(1)
            df = df.drop(columns=x_cols)

            # After shifting, drop the first row (which will have NaN in the lag)
            df = df.iloc[1:]


        # Optional numeric NA handling
        if self.fillna_numeric is not None:
            num_cols = df.select_dtypes(include=["number"]).columns
            df[num_cols] = df[num_cols].fillna(self.fillna_numeric)

        # Store and optionally split
        self.df = df

        if self.return_X_y:
            if self.target_col not in df.columns:
                raise KeyError(f"Target column '{self.target_col}' not found for X/y split.")
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]
            self.X_, self.y_ = X, y
            return X, y  # type: ignore[return-value]

        return df
