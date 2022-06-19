from cgi import test
from msilib import sequence
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_events_summary(PATH: str, do_plot: bool = False) -> pd.DataFrame:
    """
    Collects "eventsData.csv" files from parent directory and joins all of them into one pd.DataFrame

    """
    wells_ = [1, 2, 3, 6, 7, 8]

    files = {}
    events_summary = pd.DataFrame()

    for i in wells_:
        path = PATH + "\\" + str(i)

        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                if filename in files:
                    files[filename] += 1
                else:
                    files[filename] = 1

                # Collecting events data
                if filename in ["eventsData1.csv", "eventsData.csv"]:
                    if filename in files:
                        events = pd.read_csv(
                            os.path.join(dirname, filename),
                            parse_dates=["startDate", "endDate"],
                        )
                        events = events[(events["event"] != "Апробация правила")]
                        events["losses"] = events["losses"].apply(
                            lambda x: np.float16(x.replace(",", "."))
                        )
                        events["well_id"] = i

                    events_summary = pd.concat([events_summary, events], axis=0)

    if do_plot:
        events_summary_plot = events_summary.copy()
        events_summary_plot["month_name"] = events_summary_plot[
            "startDate"
        ].dt.month_name()
        events_summary_plot["month_year"] = (
            events_summary_plot["month_name"]
            + "_"
            + events_summary_plot["startDate"].dt.year.astype(str)
        )
        events_summary_plot.sort_values(by="startDate", inplace=True)
        month_group = events_summary_plot.groupby(
            ["month_year", "event"], as_index=False, sort=False
        )["id"].count()

        plt.figure(figsize=(15, 10))
        sns.barplot(x="month_year", y="id", hue="event", data=month_group, alpha=0.7)
        plt.legend(loc="upper right")
        plt.ylabel("Количество осложнений", fontdict={"size": 12})
        plt.xlabel("Месяц", fontdict={"size": 12})
        ax = plt.gca()
        ax.tick_params(axis="both", which="major", labelsize=12)
        plt.xticks(rotation=30)
        plt.show()

    return events_summary


def join_events_to_data(
    data_in: pd.DataFrame, events_in=None, well_id=None
) -> pd.DataFrame:
    """
    This function assingns multiple events as marks into dataset based on startDate and endDate in events dataframe

    """
    out_data = data_in.copy()

    if well_id is None:
        well_id = out_data["well"].values[0]

    if events_in is None:
        PATH = "C:\\Users\\vladv\\predictiveAnalytics\\data\\"
        events_in = get_events_summary(PATH)

    events = events_in.copy()
    events["startDate"] = pd.to_datetime(events["startDate"])
    events["endDate"] = pd.to_datetime(events["endDate"])
    events = events[events["well_id"] == int(well_id)]

    out_data["time"] = pd.to_datetime(out_data["time"])
    out_data = out_data.set_index("time")
    events_dates = events[["startDate", "endDate"]].values
    events_id = events["result"].values
    out_data["event_id"] = 0

    for ev_id, (start_date, end_date) in zip(events_id, events_dates):
        mask = (out_data.index >= start_date) & (out_data.index <= end_date)
        out_data.loc[mask, "event_id"] = ev_id

    out_data = out_data.reset_index()

    return out_data


# Should be used after features renaming
def join_data_v2(PATH: str, resample=True, verbose=False) -> pd.DataFrame:
    """
    Takes all .csv files with 2 columns in passed directory and joins them all into one pd.DataFrame based on 'time' index

    All features are resampled with 2 min and interpolated in order to have smooth data without gaps

    Should be used after features renaming

    """
    # TODO -> 1) create folders structure to save intermediate results
    # 2) Should be placed in common pipeline as step 2 in data preprocessing

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

        csv_file = (
            pd.read_csv(path, parse_dates=["time"])
            .drop_duplicates()
            .sort_values(by="time")
        )
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

    # TODO -> folders structure right here!!!
    full_data_pivot.to_csv(PATH + "\\full_data.csv")
    full_data_events.to_csv(PATH + "\\full_data_events.csv")

    return full_data_events


def reduce_mem_usage(df: pd.DataFrame, verbose=True):
    """
    Reduces pd.DataFrame memory usage based on columns types

    """
    # TODO -> think about its place in the common pipeline

    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )

    return df


def check_nans(list_wells: list):
    """
    Literally checks NaNs in list of pd.DataFrame's
    Prints amount of NaNs by column

    """
    for well_data in list_wells:
        well_id = well_data["well"].unique()[0]
        cols = well_data.columns
        total_nans = well_data.isna().sum().sum()

        if total_nans:
            print(f"Total nans of well {well_id} : {total_nans}")
            for col in cols:
                nans = well_data[col].isna().sum()
                if nans:
                    print(f"Feature {col} nans count : {nans}")


def split_data(data_in: pd.DataFrame) -> pd.DataFrame:
    """
    Function filters data based on adequancy of the data
    Based on vizualization and my personal experience 
    some data looks bad due to mistakes, gaps and working regime differences

    """
    data = data_in.copy()

    # This dict contains *from [day, month] *to [day, month] ranges of valid data based on vizualization
    wells_ranges = {
        1: [[[15, 4], [1, 9]], [[11, 10], [31, 12]]],
        2: [[[1, 5], [1, 9]], [[16, 11], [31, 12]]],
        3: [[[1, 7], [31, 12]]],
        4: [[[1, 5], [31, 12]]],  # motor temperature is constant (impossible :( )
        6: [[[1, 5], [31, 12]]],
    }

    well_id = data["well"].values[0].astype(int)

    # This data is alright
    if well_id in [7, 8]:
        return data

    data_ranges = wells_ranges[well_id]

    data_out = pd.DataFrame(columns=data.columns)
    for dates in data_ranges:
        if data.index.dtype == np.int64:
            data_temp = data.loc[
                (
                    data["time"].dt.date
                    >= pd.Timestamp(f"2021-{dates[0][1]}-{dates[0][0]}")
                )
                & (
                    data["time"].dt.date
                    <= pd.Timestamp(f"2021-{dates[1][1]}-{dates[1][0]}")
                )
            ]
        else:
            data_temp = data.loc[
                (
                    data.index.to_series().dt.date
                    >= pd.Timestamp(f"2021-{dates[0][1]}-{dates[0][0]}")
                )
                & (
                    data.index.to_series().dt.date
                    <= pd.Timestamp(f"2021-{dates[1][1]}-{dates[1][0]}")
                )
            ]
        data_out = pd.concat([data_out, data_temp], axis=0)

    return data_out


def del_z_outliers(data_in: pd.DataFrame) -> pd.DataFrame:
    """
    Easiest approach to filter mistakes and outcasts in data
    Uses 3 sigma rule to find outcasts and filters 
    indecies where the observation is out of 3 std's

    """
    data = data_in.copy()
    # data = data_out[data_out["event_id"] == 0] Should I keep it?

    if data_in.index.dtype != np.int64:
        raise ValueError("Index type is not correct")

    data = data.reset_index(drop=True)
    cols = data.columns
    cols = [
        col
        for col in cols
        if col
        not in [
            "well",
            "event_id",
            "time",
            "event_before",
            "failure_target",
            "stable",
            "time_to_failure",
        ]
    ]
    target_indices = data[data["failure_target"] == 1].index.tolist()
    shape_before = data.shape[0]
    out = []

    for col in cols:
        series_to_check = data[col]
        m = np.mean(series_to_check)
        sd = np.std(series_to_check)

        try:
            for i, value in enumerate(series_to_check):
                z = (value - m) / sd
                if np.abs(z) > 3 and i not in target_indices:
                    out.append(i)
        except ZeroDivisionError:
            print(f"Feature {col} is constant")
            continue

    out = set(out)
    data = data.drop(out, axis=0)
    shape_after = data.shape[0]
    print(f"Initial shape : {shape_before}, outliers : {shape_before-shape_after}")

    return data


# TODO -> Donot remove the correlated fetures but use PCA to get only one out of them for each highly correlated group
# TODO -> Обработка сигналов, какие преобразования еще можно ебнуть для токов
def features_calculation(data_in: pd.DataFrame, cols_to_calc=None):
    """ "
    Calculating unbalances, differentiables, statistics of operating parameters as new features

    """
    # TODO -> 1) Place function in the common pipeline
    # 2) Define and analyze all the statistics that is most suitable for prediction and logic

    data = data_in.copy()
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

    data["current_unbalance"] = data["current_unbalance"].fillna(
        0
    )  # Impute zeros where currents are zeros
    data["voltage_unbalance"] = data["voltage_unbalance"].fillna(
        0
    )  # Impute zeros where voltages are zeros

    # I don't need currents anymore, because active power represent variability of all currents (based on correlation matrix)
    data["voltage"] = data[
        "voltageAB"
    ]  # Lets keep only one voltage to save variability
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
            "motor_load",
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
    # cols_to_drop = [col for col in initial_cols if col not in ["well", "event_id", "time"]]
    # cols_to_drop.extend(["voltage", "current"])
    data = data.drop(cols_to_drop, axis=1)

    # data = del_z_outliers(data)

    return data


def check_oil_rate(
    data_in: pd.DataFrame, drop=False, impute=False, plot=False
) -> pd.DataFrame:
    """
    This function checks if oilrate higher than liquid rate.
    Depending on results we can drop oilrate column or impute liquid rate instead wrong values

    """
    data_out = data_in.copy()
    cols = data_out.columns

    # Some troubles with features names
    try:
        well_id = data_out["well"].values[0]

        oil_rate = data_out["oil_rate"]
        liquid_rate = data_out["liquid_rate"]
        wrong_values = oil_rate > liquid_rate

        wrong_sum = wrong_values.sum()

        if wrong_sum > 0:

            print(
                f"Data has {wrong_sum} incorrect values of Oil Rate in Well {well_id}"
            )

            if drop and impute:
                raise ValueError("Both arguments True")

            if plot:
                plt.figure(figsize=(15, 8))
                plt.plot(oil_rate, label="Oil Rate", alpha=0.75)
                plt.plot(liquid_rate, label="Liquid Rate", alpha=0.75)
                plt.xlabel("Date")
                plt.ylabel("Rate")
                plt.title(f"Oil and Liquid Rates of {well_id} well")
                plt.legend()
                plt.grid()
                plt.show()

            if drop:
                data_out = data_out[data_out["oil_rate"] <= data_out["liquid_rate"]]

                return data_out

            if impute:
                mask = data_out["oil_rate"] > data_out["liquid_rate"]
                data_out.loc[mask, "oil_rate"] = data_out.loc[mask, "liquid_rate"]

                return data_out
    except KeyError:
        print(f"Well {well_id} doesn't have oil rate records")


# Machines are engineered to last. If something breaks all the time, you won’t buy it, would you?
# Because machines generally last a long time, we typically do not have many examples of failure.
# This means the data sets we use in PM are almost always unbalanced.


def expand_target(data_in: pd.DataFrame, target_window=7, split=False) -> pd.DataFrame:
    """
    Based on 'target_window' creates additional variable as target in advance to failure
    For example if 'target_window'=7 -- all observations (rows) before failure in 7 days are signed as class '1'

    """
    data = data_in.copy()

    if data.index.dtype != np.int64:
        data = data.reset_index()

    # data = data[(data["event_id"] == 966) | (data["event_id"] == 266)]

    data["time"] = pd.to_datetime(data["time"])
    data = data.sort_values(by="time", ascending=True)
    data.loc[(data["event_id"] != 0), "failure_date"] = data.loc[
        (data["event_id"] != 0), "time"
    ]
    data["failure_date"] = data["failure_date"].bfill().fillna(data["time"].max())
    data["failure_date"] = pd.to_datetime(data["failure_date"])

    data["fail_range"] = data["failure_date"] - data["time"]
    data["time_to_failure"] = data["fail_range"] / np.timedelta64(1, "D")
    data.loc[data["failure_date"] == data["time"].max(), "time_to_failure"] = 999

    # I use window between 7 and 3 days before failure
    data["failure_target"] = np.where(
        ((data["time_to_failure"] <= target_window) & ((data["time_to_failure"] > 3))),
        1,
        0,
    )
    data = data.drop(["failure_date", "fail_range"], axis=1)
    # data["stable"] = np.where(((data["time_to_failure"] >= 30) & (data["time_to_failure"] <= 90)), 1, 0)
    # data = data[data["time_to_failure"] != 999]
    data["stable"] = np.where((data["time_to_failure"] >= 20), 1, 0)

    if split:
        data = split_data(data)

    data = del_z_outliers(data)

    return data


def join_data(list_data: list) -> pd.DataFrame:
    """ "
    This function joins all datasets in 'list_data' into one pd.DataFrame and splits it in train and test

    """
    # TODO -> 1) Should be used in the common pipeline data preprocessing as last step
    # 2) Define the period where to test models
    all_data = list_data.copy()
    joined_df = pd.DataFrame(columns=all_data[0].columns)

    for df in list_data:
        joined_df = pd.concat([joined_df, df], axis=0)

    joined_df = joined_df.reset_index(drop=True)
    joined_df["time"] = pd.to_datetime(joined_df["time"])
    joined_df = joined_df.sort_values(by="time", ascending=True)
    joined_df = pd.concat(
        [joined_df, pd.get_dummies(joined_df["well"], prefix="Well_")], axis=1
    )
    joined_df = joined_df.drop("well", axis=1)
    joined_df = joined_df.reset_index(drop=True)

    test_df = joined_df[
        (
            (joined_df["time"].dt.date >= pd.Timestamp(f"2021-06-01"))
            & (joined_df["time"].dt.date <= pd.Timestamp(f"2021-06-30"))
        )
    ]
    print(f"Joined data shape : {joined_df.shape}")
    test_indices = test_df.index
    train_df = joined_df.drop(test_indices, axis=0)
    # train_df = train_df.loc[((train_df["stable"] == 1) & (train_df["failure_target"] == 0)) | ((train_df["failure_target"] == 1))]
    # train_df = del_z_outliers(train_df)
    test_df = test_df.drop(["stable", "time_to_failure"], axis=1)
    train_df = train_df.drop(["stable", "time_to_failure"], axis=1)

    return train_df, test_df
