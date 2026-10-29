"""Chronological splits and training-prefix scaling."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Subset


def read_data(path: str, nrows=None) -> pd.DataFrame:
    """
    Read the data file and return DataFrame.
    According to the provided file path, read the data file and return the corresponding DataFrame.
    :param path: The path to the data file.
    :return:  The DataFrame of the content of the data file.
    """
    data = pd.read_csv(path)
    label_exists = "label" in data["cols"].values

    all_points = data.shape[0]

    columns = data.columns

    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()

    is_univariate = n_points == all_points

    n_cols = all_points // n_points
    df = pd.DataFrame()

    cols_name = data["cols"].unique()

    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    else:
        df[cols_name[0]] = data.iloc[:, 0]

    if label_exists:
        # Get the column name of the last column
        last_col_name = df.columns[-1]
        # Renaming the last column as "label"
        df.rename(columns={last_col_name: "label"}, inplace=True)

    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]

    return df


class ForecastWindows(Dataset):
    def __init__(self, values, start, end, seq_len, horizon, future_mode="realized"):
        self.values = values
        self.seq_len, self.horizon = seq_len, horizon
        self.future_mode = future_mode
        self.origins = np.arange(max(start, seq_len), end - horizon + 1)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = self.origins[index]
        return (
            self.values[origin - self.seq_len : origin],
            (
                np.zeros(
                    (self.horizon, self.values.shape[1] - 1), dtype=self.values.dtype
                )
                if self.future_mode == "history"
                else self.values[origin : origin + self.horizon, 1:]
            ),
            self.values[origin : origin + self.horizon, :1],
        )


def load_windows(args):
    frame = read_data(args.data_path)
    frame = frame[[args.target] + [c for c in frame.columns if c != args.target]]
    if getattr(args, "calendar", False):
        import pandas as pd

        dates = pd.DatetimeIndex(frame.index)
        for name, phase in (
            ("hour", dates.hour / 24),
            ("weekday", dates.dayofweek / 7),
            ("annual", (dates.dayofyear - 1) / 365.25),
        ):
            frame["calendar_" + name + "_sin"] = np.sin(2 * np.pi * phase)
            frame["calendar_" + name + "_cos"] = np.cos(2 * np.pi * phase)
    # Same nested integer rounding as TFB's rolling_forecast strategy.
    tv_end = int(len(frame) * 0.8)
    train_end = int(tv_end * 0.875)
    fold = getattr(args, "validation_fold", 2)
    if fold != 2:
        if args.evaluate_test:
            raise ValueError("Earlier rolling folds cannot evaluate the official test")
        official_train_end = train_end
        train_end = int(official_train_end * (5 + fold) / 7)
        tv_end = int(official_train_end * (6 + fold) / 7)
    scaler = StandardScaler().fit(frame.iloc[:train_end].values)
    values = scaler.transform(frame.values).astype(np.float32)
    boundaries = {"train": (0, train_end), "validation": (train_end, tv_end)}
    if args.evaluate_test:
        boundaries["test"] = (tv_end, len(frame))
    datasets = {
        name: ForecastWindows(
            values,
            start,
            end,
            args.seq_len,
            args.horizon,
            getattr(args, "future_mode", "realized"),
        )
        for name, (start, end) in boundaries.items()
    }
    metadata = {
        "columns": list(frame.columns),
        "train_end": train_end,
        "validation_end": tv_end,
        "length": len(frame),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "stride": 1,
        "drop_last": False,
        "origins": {name: len(data) for name, data in datasets.items()},
        "future_mode": getattr(args, "future_mode", "realized"),
    }
    if getattr(args, "lag_grid", "fixed") == "auto":
        import pandas as pd

        spacing = np.median(np.diff(pd.DatetimeIndex(frame.index).asi8)) / 3.6e12
        periods = (24, 168, 720, 1440) if spacing >= 24 else (24, 168)
        metadata["lag_candidates"] = sorted(
            set(
                [0, 1, 3, 6, 12, 24]
                + [max(1, round(hours / spacing)) for hours in periods]
            )
        )
    return datasets, metadata


def subset(data, count):
    if count >= len(data):
        return data
    return Subset(data, np.linspace(0, len(data) - 1, count, dtype=int))
