import pandas as pd
import numpy as np
from tqdm import tqdm
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

RAW = "wifiData.csv"
DES = "wifiData_deseasonalised.csv"
FREQ = "10min"
TRAIN_FRAC = 0.70          # train split; the profiles themselves are fitted upstream
T = 48                     # context length
deltas = [1]
MODEL_PATH = "amazon/chronos-bolt-tiny"


def seasonal_component(raw_path, des_path):

    raw = pd.read_csv(raw_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    des = pd.read_csv(des_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if len(raw) != len(des) or not (raw.date.values == des.date.values).all():
        raise SystemExit(f"{raw_path} and {des_path} are not aligned -- regenerate "
                         f"the deseasonalised file with plot_deseasonalizeWifi.py")
    x = raw.OT.to_numpy(dtype=float)
    return x - des.OT.to_numpy(dtype=float), x


season, x_raw = seasonal_component(RAW, DES)

variants = [
    ("raw", RAW, False),
    ("deseasonalized", DES, True),
]

for delta in deltas:
    for fine_tune in [False, True]:
        tag = "CHRNS_TUNED" if fine_tune else "CHRNS_ZERO"
        for series_name, filename, reseasonalize in variants:
            print("\n============================================================")
            print(f"{tag} | delta={delta} | {series_name} ({filename})")
            print("============================================================\n")

            # 1. Load the data
            df = pd.read_csv(filename)
            if "Active_Users" in df.columns:
                df = df.drop(columns=["Active_Users"])

            df["item_id"] = "Series_1"
            df["date"] = pd.to_datetime(df["date"])

            # 2. Format and enforce frequency
            ts_df = TimeSeriesDataFrame.from_data_frame(
                df,
                id_column="item_id",
                timestamp_column="date"
            )
            ts_df = ts_df.convert_frequency(freq=FREQ)

            # 3. 70% train / 10% held out (unused) / last 20% test
            total_length = len(ts_df)
            train_length = int(total_length * TRAIN_FRAC)
            test_length = int(total_length * 0.20)
            test_start = total_length - test_length

            print(f"Total steps: {total_length} | Train: {train_length} | "
                  f"Unused: {test_start - train_length} | Test: {test_length}")

            # 4. Standardize using ONLY training statistics (no data leakage)
            train_mean = ts_df["OT"].iloc[:train_length].mean()
            train_std = ts_df["OT"].iloc[:train_length].std()
            ts_df["OT"] = (ts_df["OT"] - train_mean) / train_std

            ot_values = ts_df["OT"].values
            timestamps = ts_df.index.get_level_values("timestamp")

            # 5. Fit (fine-tune) or just load (zero-shot) Chronos-Bolt on the 80% train split
            train_data = ts_df.iloc[:train_length]

            print(f"\n--- Chronos-Bolt setup (fine_tune={fine_tune}) ---")
            predictor = TimeSeriesPredictor(
                target="OT",
                prediction_length=delta,
                freq=FREQ
            ).fit(
                train_data,
                hyperparameters={
                    "Chronos": {
                        "model_path": MODEL_PATH,
                        "fine_tune": fine_tune,
                        "context_length": T,
                    },
                },
                verbosity=4
            )

            # 6. Build batched context windows from the NORMALIZED ts_df
            print("\n--- Preparing Batched Data for Delta-Step Forecasts ---")
            CONTEXT_LENGTH = T

            batched_data = []
            valid_test_length = test_length - delta + 1

            for i in tqdm(range(valid_test_length), desc=f"Slicing windows ({series_name})"):
                current_step_index = test_start + i

                start_index = max(0, current_step_index - CONTEXT_LENGTH)
                window_df = pd.DataFrame({
                    "OT": ot_values[start_index:current_step_index],
                    "date": timestamps[start_index:current_step_index],
                    "item_id": f"W_{i}"
                })
                batched_data.append(window_df)

            batched_df = pd.concat(batched_data, ignore_index=True)
            batched_ts_df = TimeSeriesDataFrame.from_data_frame(
                batched_df,
                id_column="item_id",
                timestamp_column="date"
            )
            batched_ts_df = batched_ts_df.convert_frequency(freq=FREQ)

            print("\n--- Running Batched GPU Inference ---")
            predictions = predictor.predict(batched_ts_df)

            # 7. Predictions, back in the file's own units
            y_pred = np.array([
                predictions.loc[f"W_{i}"]["mean"].values[:delta]
                for i in range(valid_test_length)
            ])
            y_pred = y_pred * train_std + train_mean

            # 8. Absolute row index of every forecast target, shape (windows, delta)
            idx = (test_start + np.arange(valid_test_length))[:, None] + np.arange(delta)[None, :]

            # Add the removed component back so both variants live in raw space
            if reseasonalize:
                y_pred = y_pred + season[idx]

            # 9. Always score against the RAW ground truth
            y_true = x_raw[idx]

            # MASE per horizon, then averaged over the horizons -- same shape as
            # the NMSE it replaces. The naive scale is the mean absolute
            # one-step change of that horizon's own truth column, so each
            # horizon is normalised by the series it is scored against and 1.0
            # is the persistence forecast.
            mase_per_horizon = []
            for h in range(delta):
                mae_h = np.mean(np.abs(y_true[:, h] - y_pred[:, h]))
                scale_h = np.mean(np.abs(np.diff(y_true[:, h])))
                mase_per_horizon.append(mae_h / scale_h)

            # Per-forecast dump for wifiData_flows/compare_forecasts.py, which
            # scores every forecaster on one window with one denominator and
            # reports MSE as well -- neither of which res.txt can do. Only at
            # delta = 1: past that a window carries `delta` forecasts and the
            # dump schema is one forecast per bucket, so there is no honest way
            # to write the multi-horizon runs into it.
            #
            # `season` is the component ADDED BACK (zero for the raw variant).
            # compare_forecasts.py checks it against deseason.py, which is the
            # only check that catches a stale deseasonalised csv: such a file
            # keeps its dates and row count, so the alignment assert in
            # seasonal_component passes right through it.
            if delta == 1:
                flat = idx[:, 0]
                pd.DataFrame({
                    "idx": flat,
                    "date": timestamps.to_numpy()[flat],
                    "truth": y_true[:, 0],
                    "season": season[flat] if reseasonalize else np.zeros(len(flat)),
                    tag: y_pred[:, 0],
                }).to_csv(f"./fc_{tag.lower()}_{series_name}.csv", index=False)
                print(f"per-forecast values -> ./fc_{tag.lower()}_{series_name}.csv")

            mean_mase = np.mean(mase_per_horizon)
            print(f"\n{tag} delta={delta} {series_name}: MASE = {mean_mase:.4f}")

            with open("res.txt", "a") as f:
                f.write(f"{tag:<12}{T:>6}{RAW:>18}{series_name:>16}{mean_mase:>10.4f}\n")
