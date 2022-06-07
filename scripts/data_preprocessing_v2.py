import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer



class preprocessing_pipeline:
    
    def __init__(self, directory):
        self.directory = directory

    def _rename_features(self, input_data, columns, feature_name):
        data_in = input_data.copy()

        new_name = ""
        for i, char in enumerate(feature_name):
            if char.isupper() and not (char == feature_name[-2] or char == feature_name[-1]):
                if i == 0:
                    new_name += char.lower()
                else:
                    new_name += "_" + char.lower()
            else:
                new_name += char

        data_in = data_in.rename(columns={columns[0]: "time", columns[1]: new_name})
        data_in["time"] = pd.to_datetime(data_in["time"])
        data_in = data_in.sort_values(by="time")
        data_in = data_in.reset_index(drop=True)
        data_in = data_in.set_index("time")

        return data_in

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
        events = get_events_summary(PATH[:-3])
        full_data_events = join_events_to_data(full_data_pivot, events, well_id)

        # full_data_pivot.to_csv(PATH+"\\full_data.csv")
        # full_data_events.to_csv(PATH+"\\full_data_events.csv")

        return full_data_events

    def data_import(self, do_return=False, verbose=True):
        pass

    def data_process(self):
        pass

