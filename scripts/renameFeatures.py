from cProfile import label
import csv
from functools import total_ordering
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits import mplot3d
from yaml import parse

#TODO packages


def rename_features(PATH, verbose=False):

    for dirname, _, filenames in os.walk(PATH):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)

            if verbose:
                print(file_path)
            try:
                if file_path[-4:] == ".csv":
                    csv_file = pd.read_csv(file_path)
                    cols = csv_file.columns

                    # Data needed is assumed to have only 2 columns: vUpdateTime and value
                    if len(cols) != 2:
                        if verbose:
                            print(f"Cols more than 2 : {len(cols)}")
                        raise KeyError

                    feature_name = filename[:-4]
                    new_name = ""
                    for i, char in enumerate(feature_name):
                        if char.isupper() and not (char == feature_name[-2] or char == feature_name[-1]):
                            if i == 0:
                                new_name += char.lower()
                            else:
                                new_name += "_" + char.lower()
                        else:
                            new_name += char

                    csv_file.rename(columns={cols[0]: "time", cols[1]: new_name}, inplace=True)
                    csv_file["time"] = pd.to_datetime(csv_file["time"])
                    csv_file.sort_values(by="time", inplace=True)
                    csv_file.reset_index(drop=True, inplace=True)
                    csv_file.set_index("time", inplace=True)

                    csv_file.to_csv(file_path)

                    if verbose:
                        print(f"Features of {filename} renamed")

            except KeyError:
                if verbose:
                    print(f"Skipping file : {filename}")
                continue


def get_data_summary(PATH: str=None, data_in: pd.DataFrame=None) -> pd.DataFrame:
    """"
    In case 'PATH' is given: reads all the csv files in directory with 2 columns (time, value) and calculates summary
    In case 'data_in' is given: calculates summary for all presented columns
    
    """

    if PATH and data_in:
        raise KeyError("Function takes either PATH on pd.DataFrame but both were given")

    summary_cols = [
        "file_name", "total_rows", "nan_rows", "zero_rows", "duplicates", 
        "unique_values", "mean_value", "mean_shift", "max_shift", "max_shift_from", "min_shift", "min_date", "max_date"
    ]
    summary = pd.DataFrame(columns=summary_cols)

    if PATH:

        well = PATH[-1]
        for dirname, _, filenames in os.walk(PATH):
            for ix, filename in enumerate(filenames):

                try:
                    path = os.path.join(dirname, filename)
                    file_summary = pd.DataFrame(columns=summary_cols)
                    csv_file = pd.read_csv(path)
                    columns = csv_file.columns

                    # Skipping non relevant data
                    if len(columns) != 2:
                        raise KeyError
                    
                    csv_file["time"] = pd.to_datetime(csv_file["time"])
                    csv_file.sort_values(by="time", inplace=True)
                    csv_file.reset_index(drop=True, inplace=True)
                    values = csv_file[columns[1]]

                    total_rows = csv_file.shape[0]
                    csv_file.drop_duplicates(inplace=True)
                    duplicates = total_rows - csv_file.shape[0]
                    nan_rows = values.isna().sum()
                    zero_rows = total_rows - np.count_nonzero(values)
                    unique_values = values.nunique()
                    mean_value = values.mean()

                    csv_file["diffs"] = csv_file[columns[0]].diff()
                    mean_shift = csv_file["diffs"].mean()
                    max_shift = csv_file["diffs"].max()
                    max_shift_from = csv_file[csv_file["diffs"] == max_shift][columns[0]]
                    min_shift = csv_file["diffs"].min()
                    min_date = csv_file[columns[0]].min()
                    max_date = csv_file[columns[0]].max()

                    file_summary.loc[ix] = [
                        filename[:-4], total_rows, nan_rows, zero_rows, duplicates, unique_values, mean_value, 
                        mean_shift, max_shift, max_shift_from, min_shift, min_date, max_date
                    ]    
                    summary = pd.concat([summary, file_summary], axis=0)

                except KeyError:
                    print(f"Skipping file {filename}")
                    continue

        summary["well"] = well
        save_path = PATH+"\\data_summary.csv"

        summary.to_csv(save_path)

    if data_in:
        cols_data = data_in.columns
        file_summary = pd.DataFrame(columns=summary_cols)
        csv_file = pd.read_csv(path)
        columns = csv_file.columns

        # Skipping non relevant data
        if len(columns) != 2:
            raise KeyError
        
        csv_file["time"] = pd.to_datetime(csv_file["time"])
        csv_file.sort_values(by="time", inplace=True)
        csv_file.reset_index(drop=True, inplace=True)
        values = csv_file[columns[1]]

        total_rows = csv_file.shape[0]
        csv_file.drop_duplicates(inplace=True)
        duplicates = total_rows - csv_file.shape[0]
        nan_rows = values.isna().sum()
        zero_rows = total_rows - np.count_nonzero(values)
        unique_values = values.nunique()
        mean_value = values.mean()

        csv_file["diffs"] = csv_file[columns[0]].diff()
        mean_shift = csv_file["diffs"].mean()
        max_shift = csv_file["diffs"].max()
        max_shift_from = csv_file[csv_file["diffs"] == max_shift][columns[0]]
        min_shift = csv_file["diffs"].min()
        min_date = csv_file[columns[0]].min()
        max_date = csv_file[columns[0]].max()

        file_summary.loc[ix] = [
            filename[:-4], total_rows, nan_rows, zero_rows, duplicates, unique_values, mean_value, 
            mean_shift, max_shift, max_shift_from, min_shift, min_date, max_date
        ]    
        summary = pd.concat([summary, file_summary], axis=0)


    
    return summary


# Doesn't work really well, cause gives too much values (~2kk) or skips data
def join_data(PATH, method="resample"):
    
    total_data = pd.DataFrame()

    for dirname, _, filenames in os.walk(PATH):
        for i, filename in enumerate(filenames):
            file_path = os.path.join(dirname, filename)
            try:
                csv_file = pd.read_csv(file_path)
                cols = csv_file.columns

                if len(cols) != 2:
                    print(f"Cols more than 2 : {len(cols)}")
                    raise ValueError

                if filename in ["firstHalfMotorLoad.csv", "secondHalfMotorLoad.csv"]:
                    print(f"Skipping {filename}")
                    raise ValueError
                
                csv_file["time"] = pd.to_datetime(csv_file["time"])

                csv_file.drop_duplicates(inplace=True)
                csv_file.sort_values(by="time", inplace=True)
                # csv_file.reset_index(inplace=True, drop=True)
                csv_file = csv_file[csv_file["time"].dt.year < 2022]
                csv_file.set_index("time", inplace=True)
                csv_file = csv_file[~csv_file.index.duplicated()]

                if method == "union":
                    if i == 0:
                        nidx = pd.date_range(csv_file.index.min(), csv_file.index.max(), freq="1.5Min")
                    oidx = csv_file.index
                    csv_file = csv_file.reindex(nidx.union(oidx)).interpolate("index").reindex(nidx)
                    csv_file = csv_file.reset_index().rename(columns={"index": "time"})

                if method == "resample":
                    csv_file = csv_file.resample("1.5Min").interpolate()

                if method == "nearest":
                    if i == 0:
                        nidx = pd.date_range(csv_file.index.min(), csv_file.index.max(), freq="1.5Min")
                    csv_file = csv_file.reindex(nidx, method="nearest", limit=1).interpolate()
                    csv_file = csv_file.reset_index().rename(columns={"index": "time"})

                print(f"Shape of file {filename} {csv_file.shape}, nans {csv_file.isna().sum()[0]}")

                # print(csv_file.info())
                if i == 0:
                    print(csv_file.columns)
                    total_data = csv_file.copy()
                else:
                    total_data = pd.merge(total_data, csv_file, how="outer", on="time")
                
                print(f"File {filename} joined, shape of total {total_data.shape}")

            except ValueError:
                print(f"Skipping file : {filename}")
                continue
    
    # print(total_data.head())
    total_data.to_csv(PATH+"\\joined_data.csv")

    return total_data

    
def get_data_summary2(data_in: pd.DataFrame):

    summary_cols = ["total_rows", "nan_rows", "zero_rows", "duplicates", "unique_values", "mean_value", "mean_shift"]
    data = data_in.copy()

    # if data.index.dtype in [np.int0, np.int16, np.int32, np.int64]:
    #     data = data.set_index("time")

    data_cols = data_in.columns
    summary = pd.DataFrame(index=data_cols, columns=summary_cols)
    total_rows = data.shape[0]
    data = data.drop_duplicates()

    for col in data_cols:
        values = data[col]

        duplicates = total_rows - data.shape[0]
        nan_rows = values.isna().sum()
        zero_rows = total_rows - np.count_nonzero(values)
        unique_values = values.nunique()
        mean_value = values.mean()
        diffs = values.index.to_series().diff()
        mean_shift = diffs.mean()

        summary.loc[col] = [total_rows, nan_rows, zero_rows, duplicates, unique_values, mean_value, mean_shift]
    
    return summary
        





