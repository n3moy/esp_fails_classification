import os
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_datetime64_any_dtype as is_datetime


# This class is designed only to join raw data
class DataJoin:
    def __init__(self, directory_from, directory_to):
        self.directory_from = directory_from
        self.directory_to = directory_to
        self.renamed_files_paths = []
        self.raw_files_paths = []
        if not self.directory_to.endswith("\\"):
            self.directory_to += "\\"

    def _rename_features(
        input_data: pd.DataFrame, feature_name: str, columns: list
    ) -> pd.DataFrame:
        data_out = input_data.copy()
        new_name = ""
        for i, char in enumerate(feature_name):
            if feature_name == "voltageAC":
                new_name = "voltageCA"  # Mistake in initial data
                break
            if feature_name == "activePower1":
                new_name = "active_power"  # Mistake in initial data
                break
            if "Gage" in feature_name:
                new_name = "electricity_gage"  # Mistake in initial data
                break
            if char.isupper() and not (
                char == feature_name[-2] or char == feature_name[-1]
            ):
                if i == 0:
                    new_name += char.lower()
                else:
                    new_name += "_" + char.lower()
            else:
                new_name += char
        data_out = data_out.rename(columns={columns[0]: "time", columns[1]: new_name})
        return data_out

    def _join_events_to_data(data_in: pd.DataFrame, events_in=None) -> pd.DataFrame:
        """
        This function assingns multiple events as marks into dataset based on startDate and endDate in events dataframe

        """
        out_data = data_in.copy()
        events = events_in.copy()
        events_dates = events[["startDate", "endDate"]].values
        events_id = events["result"].values
        out_data["event_id"] = 0
        for ev_id, (start_date, end_date) in zip(events_id, events_dates):
            mask = (out_data.index >= start_date) & (out_data.index <= end_date)
            out_data.loc[mask, "event_id"] = ev_id
        out_data = out_data.reset_index()
        return out_data

    def _find_common_features(self):
        pass

    # First step in pipeline
    def data_collecting(self, do_return=False, verbose=True):
        """
        From a given directory collects all paths to files
        provided all files are in .csv format and have only 2 columns

        if do_return is set 'True' returns all paths collected
        """
        dirname, folders, _ = next(os.walk(self.directory_from))
        for folder in folders:
            if verbose:
                print(f"Collecting data from folder {folder}")
            folder_path = os.path.join(dirname, folder)
            _, __, filenames = next(os.walk(folder_path))

            # This tuple has 3 items:
            # 1) folder (aka well number),
            # 2) filenames (aka op. parameters) in folder
            # 3) complete paths for files inside folder
            self.raw_files_paths.append(
                (
                    folder,
                    filenames,
                    [os.path.join(folder_path, filename) for filename in filenames],
                )
            )
        self.raw_files_paths = sorted(self.raw_files_paths, key=lambda x: int(x[0]))
        if do_return:
            return self.raw_files_paths

    # Second step in pipeline
    # Takes about 4 mins to run 8 wells
    def rename_features(self, verbose=False):
        """
        From directory 'self.raw_files_paths' imports all the files containing 2 columns,
        renames 'value' as <filename> and 'vUpdateTime' as 'time'

        Saves all renamed files into './data/.../renamed'

        """
        for folder, filenames, filepaths in self.raw_files_paths:
            tmp_filenames = []
            tmp_filenames_paths = []
            for filename, filepath in zip(filenames, filepaths):
                csv_file = pd.read_csv(filepath)
                cols = csv_file.columns
                if len(cols) != 2:
                    if verbose:
                        print(f"Skipping file {filepath}\nFile has more than 2 columns")
                    continue

                csv_file = DataJoin._rename_features(
                    input_data=csv_file,
                    feature_name=filename.split(".")[0],
                    columns=cols,
                )
                path_to_save = (
                    self.directory_to + "renamed\\" + folder + "\\" + filename
                )

                if verbose:
                    print(f"Saving a file {filename} into directory\n{path_to_save}")

                if not os.path.exists(path_to_save[: -len(filename) - 1]):
                    os.makedirs(path_to_save[: -len(filename) - 1])
                csv_file.to_csv(path_to_save, index=False)
                tmp_filenames.append(filename)
                tmp_filenames_paths.append(path_to_save)
            # This tuple has 3 items:
            # 1) folder (aka well number),
            # 2) filenames (aka op. parameters) in folder
            # 3) complete paths for files inside folder
            self.renamed_files_paths.append(
                (folder, tmp_filenames, tmp_filenames_paths)
            )

    # Third step in pipeline
    # Around 7 mins to calculate on 8 wells
    def join_data_by_well(self, resample=True):
        for folder, filenames, filepaths in self.renamed_files_paths:
            print(f"Starting to join well #{folder}")
            full_data = pd.DataFrame(columns=["time", "value", "feature"])

            for filename, filepath in zip(filenames, filepaths):
                csv_file = pd.read_csv(filepath)
                cols = csv_file.columns
                if len(cols) != 2:
                    continue
                if "time" not in cols:
                    print(f"{filename} doesn't have 'time' column")
                    continue
                if not is_datetime(csv_file["time"]):
                    csv_file["time"] = pd.to_datetime(csv_file["time"])
                feature_name = cols[1]
                csv_file = csv_file.rename(columns={feature_name: "value"})
                csv_file["feature"] = feature_name
                csv_file = csv_file.sort_values(by="time")
                full_data = full_data.append(csv_file)
            duplicates_indices = full_data.duplicated(["time", "feature"], keep="first")
            full_data = full_data[~duplicates_indices]
            full_data_pivot = full_data.pivot(
                index="time", columns="feature", values="value"
            )
            full_data_pivot = full_data_pivot.ffill()
            well_id = folder
            if resample:
                print("Resampling...")

                # TODO -> Make different funcs for resampling
                full_data_pivot = full_data_pivot.resample("2Min").interpolate()
            print("Joining events to dataset...")
            events_path = self.raw_files_paths[int(folder) - 1][2]
            for f in events_path:
                if f.endswith("eventsData1.csv") or f.endswith("eventsData.csv"):
                    events = pd.read_csv(f, parse_dates=["startDate", "endDate"])
                    break
            full_data_pivot = DataJoin._join_events_to_data(full_data_pivot, events)
            path_to_save = self.directory_to + "joined_resampled\\"
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            full_data_pivot["well_id"] = well_id
            full_data_pivot.to_csv(
                path_to_save + f"\\joined_data_{folder}.csv", index=False
            )
            print(f"Well #{folder} is saved to {path_to_save}")


class FeatureCalculation:
    def __init__(
        self,
        directory_from: str = None,
        directory_to: str = None,
        input_data: pd.DataFrame = None,
    ):
        self.directory_from = directory_from
        self.directory_to = directory_to
        self.raw_files_paths = []
        self.input_data = input_data
        if directory_to is None or directory_from is None:
            self.DO_RETURN = True
        else:
            self.DO_RETURN = False
        if not self.directory_to.endswith("\\"):
            self.directory_to += "\\"

    def _find_common_features(self, data_lst: list) -> list:
        for df in data_lst:
            cols = df.columns
            common_cols = np.intersect1d(common_cols, cols)
        return common_cols

    def _data_collecting(self, verbose=True):
        """
        From a given directory collects all paths to files
        provided all files are in .csv format and have only 2 columns

        if do_return is set 'True' returns all paths collected
        """
        dirname, folders, _ = next(os.walk(self.directory_from))
        for folder in folders:
            if verbose:
                print(f"Collecting data from folder {folder}")
            folder_path = os.path.join(dirname, folder)
            _, __, filenames = next(os.walk(folder_path))

            # This tuple has 3 items:
            # 1) folder (aka well number),
            # 2) filenames (aka op. parameters) in folder
            # 3) complete paths for files inside folder
            self.raw_files_paths.append(
                (
                    folder,
                    filenames,
                    [os.path.join(folder_path, filename) for filename in filenames],
                )
            )
        self.raw_files_paths = sorted(self.raw_files_paths, key=lambda x: int(x[0]))

    def _features_calculation(
        input_data: pd.DataFrame, cols_to_calc: list = None
    ) -> pd.DataFrame:
        """
        Calculating unbalances, derivarives, min, max, mean, std, spike
        of operating parameters with window as new features

        """
        # TODO -> 1) Place function in the common pipeline
        # 2) Define and analyze all the statistics that is most suitable for prediction and logic
        data = input_data.copy()
        initial_cols = data.columns
        # Current and voltage unbalance
        voltage_names = ["voltageAB", "voltageBC", "voltageCA"]
        current_names = ["op_current1", "op_current2", "op_current3"]
        voltages = data[voltage_names]
        currents = data[current_names]
        mean_voltage = voltages.mean(axis=1)
        mean_current = currents.mean(axis=1)
        deviation_voltage = voltages.sub(mean_voltage, axis=0).abs()
        deviation_current = currents.sub(mean_current, axis=0).abs()
        data["voltage_unbalance"] = (
            deviation_voltage.max(axis=1).div(mean_voltage, axis=0) * 100
        )
        data["current_unbalance"] = (
            deviation_current.max(axis=1).div(mean_current, axis=0) * 100
        )
        # Impute zeros where currents are zeros
        data["current_unbalance"] = data["current_unbalance"].fillna(0)
        # Impute zeros where voltages are zeros
        data["voltage_unbalance"] = data["voltage_unbalance"].fillna(0)
        # I don't need currents anymore cause active power present variability
        # Lets keep only one voltage and current to save variability
        data["voltage"] = data["voltageAB"]
        data["current"] = data["op_current1"]
        data["resistance"] = np.where(
            (data["current"] == 0), 0, data["voltage"].div(data["current"], axis=0)
        )
        # Calculating derivatives and statictics
        if cols_to_calc is None:
            cols_to_calc = [
                "current",
                "voltage",
                "active_power",
                "frequency",
                "electricity_gage",
                # "motor_load",         # Mistake in initial data. Should be resolved some day
                "pump_temperature",
            ]
        for col in cols_to_calc:
            data[f"{col}_deriv"] = pd.Series(np.gradient(data[col]), data.index)
            data[f"{col}_rol_mean"] = (
                data[col].rolling(min_periods=1, window=60 * 14 * 3).mean()
            )
            data[f"{col}_rol_std"] = (
                data[col].rolling(min_periods=1, window=60 * 14 * 3).std()
            )
            data[f"{col}_rol_max"] = (
                data[col].rolling(min_periods=1, window=60 * 14 * 3).max()
            )
            data[f"{col}_rol_min"] = (
                data[col].rolling(min_periods=1, window=60 * 14 * 3).min()
            )
            data[f"{col}_spk"] = np.where(
                (data[f"{col}_rol_mean"] == 0), 0, data[col] / data[f"{col}_rol_mean"]
            )
            data[col] = data[col].rolling(min_periods=1, window=30).mean()
        cols_to_drop = [
            "reagent_rate",
            "oil_rate",
            "gas_rate",
            "motor_temperature",
            *voltage_names,
            *current_names,
        ]
        cols_to_drop = [col for col in cols_to_drop.copy() if col in initial_cols]
        data = data.drop(cols_to_drop, axis=1)
        return data

    def feature_calculation(self):
        data_lst = []
        common_cols = []
        for ix, filename in enumerate(os.listdir(self.directory_from)):
            print(f"Collecting data {filename}...")
            path = os.path.join(self.directory_from, filename)
            joined_file = pd.read_csv(path, parse_dates=["time"]).sort_values(by="time")
            data_lst.append(joined_file)
            cols = joined_file.columns
            if ix == 0:
                common_cols = cols
            common_cols = np.intersect1d(common_cols, cols)
        path_to_save = self.directory_to + "joined_featured"
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        for joined_data in data_lst:
            try:
                well_id = joined_data["well"].values[0]
            except KeyError:
                well_id = joined_data["well_id"].values[0]
            print(f"Calculating new features for Well #{well_id}...")
            out_data = joined_data[common_cols].copy()
            out_data = FeatureCalculation._features_calculation(out_data)
            tmp_path_to_save = path_to_save + "\\" + f"joined_featured_{well_id}.csv"
            out_data.to_csv(tmp_path_to_save, index=False)
            print(f"Output data is successfully saved to '{tmp_path_to_save}' !")
