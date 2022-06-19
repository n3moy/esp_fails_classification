import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_datetime64_any_dtype as is_datetime


class Link(list):
    def __init__(self, saved_directory):
        self.saved_directory = saved_directory


class preprocessing_pipeline:
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
        # events["startDate"] = pd.to_datetime(events["startDate"])
        # events["endDate"] = pd.to_datetime(events["endDate"])
        # out_data["time"] = pd.to_datetime(out_data["time"])
        # out_data = out_data.set_index("time")
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

                csv_file = preprocessing_pipeline._rename_features(
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
                full_data_pivot = full_data_pivot.resample("2Min").interpolate()
            print("Joining events to dataset...")
            events_path = self.raw_files_paths[int(folder) - 1][2]
            for f in events_path:
                if f.endswith("eventsData1.csv") or f.endswith("eventsData.csv"):
                    events = pd.read_csv(f, parse_dates=["startDate", "endDate"])
                    break
            full_data_pivot = preprocessing_pipeline._join_events_to_data(
                full_data_pivot, events
            )
            path_to_save = self.directory_to + "joined_resampled\\"
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            full_data_pivot["well_id"] = well_id
            full_data_pivot.to_csv(
                path_to_save + f"\\joined_data_{folder}.csv", index=False
            )
            print(f"Well #{folder} is saved to {path_to_save}")

    def find_common_features(self):
        pass

    def feature_engineering(self):
        pass

    def expand_target(self):
        pass

    def del_z_outliers(input_data: pd.DataFrame) -> pd.DataFrame:
        pass

    def make_complete_dataset():
        pass


class feature_engineering:
    pass
