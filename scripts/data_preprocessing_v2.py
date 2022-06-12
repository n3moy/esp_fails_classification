from fileinput import filename
import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer


class preprocessing_pipeline:
    
    def __init__(self, directory_from, directory_to):

        self.directory_from = directory_from
        self.directory_to = directory_to


    # @classmethod
    def _rename_features(input_data: pd.DataFrame, feature_name: str, columns: list) -> pd.DataFrame:

        data_out = input_data.copy()
        new_name = ""
        for i, char in enumerate(feature_name):
            if char.isupper() and not (char == feature_name[-2] or char == feature_name[-1]):
                if i == 0:
                    new_name += char.lower()
                else:
                    new_name += "_" + char.lower()
            else:
                new_name += char
        data_out = data_out.rename(columns={columns[0]: "time", columns[1]: new_name})

        return data_out


    def _rescale_data(self, path, verbose, resample):

        PATH = path
        files = {}
        for dirname, _, filenames in os.walk(PATH):
            for filename in filenames:
                filepath = os.path.join(dirname, filename)
                csv_file = pd.read_csv(filepath)
                cols = csv_file.columns

                if len(cols) != 2:
                    continue

                if "time" not in cols:
                    print(f"{filename} doesn't have 'time' column")

                files[filepath] = csv_file.shape[0]
        
        files = sorted(files.items(), key=lambda x: x[1], reverse=True)
        full_data = pd.DataFrame(columns=["time", "value", "feature"])

        for path, shape in files:
            csv_file = pd.read_csv(path, parse_dates=["time"]).drop_duplicates().sort_values(by="time")
            cols = csv_file.columns
            feature_name = cols[1]

            if verbose:
                print(f"Working on {feature_name}")

            csv_file = csv_file.rename(columns={feature_name: "value"})
            csv_file["feature"] = feature_name
            full_data = full_data.append(csv_file)
        
        duplicates = full_data.duplicated(["time", "feature"], keep=False)

        if verbose:
            print(f"Duplicates : {duplicates.sum()}")

        # I have only 2 duplicates in well 1, so I'm removing them since it's not going to make sufficient impact anyway
        full_data = full_data[~duplicates]
        full_data_pivot = full_data.pivot(index="time", columns="feature", values="value")
        full_data_pivot = full_data_pivot.ffill()
        well_id = PATH[-1]

        if resample:
            full_data_pivot = full_data_pivot.resample("2Min").interpolate()

        full_data_pivot["well"] = well_id
        # events = get_events_summary(PATH[:-3])
        # full_data_events = join_events_to_data(full_data_pivot, events, well_id)

        # full_data_pivot.to_csv(PATH+"\\full_data.csv")
        # full_data_events.to_csv(PATH+"\\full_data_events.csv")

        # return full_data_events


    def data_collecting(self, do_return=False, verbose=True):
        """
        From a given directory collects all paths to files provided all files are in .csv format and have only 2 columns
        
        if do_return is set 'True' returns all paths collected
        """

        dirname, folders, _ = next(os.walk(self.directory_from))
        self.raw_files_paths = []

        for folder in folders:
            if verbose:
                print(f"Collecting data from folder {folder}")
            folder_path = os.path.join(dirname, folder)
            _, __, filenames = next(os.walk(folder_path))

            # This tuple has 3 items: 1) folder (aka well number), 2) filenames (aka op. parameters) in folder 3) complete paths for files inside folder
            self.raw_files_paths.append((folder, filenames, [os.path.join(folder_path, filename) for filename in filenames]))
                
        if do_return:
            return self.raw_files_paths


    # Takes about 4 mins to run
    def rename_features(self, verbose=False):
        self.renamed_files_paths = []

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

                csv_file = preprocessing_pipeline._rename_features(input_data=csv_file, feature_name=filename.split(".")[0], columns=cols)
                path_to_save = self.directory_to + "renamed\\" + folder + "\\" + filename
                if verbose:
                    print(f"Saving a file {filename} into directory\n{path_to_save}")

                if not os.path.exists(path_to_save[:-len(filename)-1]):
                    os.makedirs(path_to_save[:-len(filename)-1])

                csv_file.to_csv(path_to_save)
                tmp_filenames.append(filename)
                tmp_filenames_paths.append(path_to_save)

            # This tuple has 3 items: 1) folder (aka well number), 2) filenames (aka op. parameters) in folder 3) complete paths for files inside folder
            self.renamed_files_paths.append((folder, tmp_filenames, tmp_filenames_paths))

