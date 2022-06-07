import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def curvatire_viz(PATH, wells):

    fig = plt.figure(figsize=(15, 15))

    for i, well in enumerate(wells):
        path = PATH + "\\" + str(well)

        curvature = pd.read_csv(path + "\\curvature.csv")
        
        ax = fig.add_subplot(len(wells)//2, 2, i+1, projection="3d")
        
        if well in [4, 5, 6, 7, 8]:
            curvature = curvature.iloc[:200]
            if well in [6, 8]:
                ax.plot(-curvature["dx"], curvature["dy"], -curvature["depth_abs"], linewidth=3)
            else:
                ax.plot(curvature["dx"], curvature["dy"], -curvature["depth_abs"], linewidth=3)
            ax.set_title(f"Well {i+1}")
            ax.set_zlabel("Depth, m", labelpad=7)
            plt.xticks(rotation=30)
            plt.yticks(rotation=-30)
        else: 
            ax.plot(curvature["dx"], curvature["dy"], -curvature["depth_abs"], linewidth=3)
            ax.set_title(f"Well {i+1}")
            ax.set_zlabel("Depth, m", labelpad=7)
            plt.xticks(rotation=30)
            plt.yticks(rotation=-30)

    # fig.tight_layout(w_pad=2)
    plt.show()


def metrics_plotting(data_in: pd.DataFrame):
    well_id = data_in["well"].values[0]
    data = data_in.drop(["well", "event_id"], axis=1)
    cols = data.columns

    current = [i for i in cols if "current" in i]
    voltage = [i for i in cols if "voltage" in i]
    pressure = [i for i in cols if "pressure" in i]
    temperature = [i for i in cols if "temperature" in i]
    rates = [i for i in cols if "rate" in i]
    other = [i for i in cols if i not in [*current, *voltage, *pressure, *temperature, *rates]]

    groups = [current, voltage, pressure, temperature, rates, other]

    # for group in groups:
    #     fig = plt.figure(figsize=(20, 15))

    #     for j, group_series in enumerate(group):
    #         ax = fig.add_subplot(len(group), 1, j+1)
    #         ax.plot(data[group_series].rolling(4, min_periods=1).mean())
    #         ax.set_title(f"{group_series} in Well {well_id}")
    #         ax.set_xlabel("Date")
    #         ax.set_ylabel(f"{group_series}")
    #         ax.grid()
    #         plt.tight_layout()

    for group in groups:

        for j, group_series in enumerate(group):
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(data[group_series].rolling(4, min_periods=1).mean())
            ax.set_title(f"{group_series} in Well {well_id}")
            ax.set_xlabel("Date")
            ax.set_ylabel(f"{group_series}")
            ax.grid()
            plt.tight_layout()


def metrics_plotting_with_events(data_in: pd.DataFrame, events_in: pd.DataFrame):
    data = data_in.copy()
    events = events_in.copy()

    if data.index.dtype == np.int64:
        data = data.set_index("time")

    well_id = data["well"].unique()[0]
    data = data.drop(["well", "event_id"], axis=1)
    cols = data.columns

    events_dates = events[["startDate", "endDate"]].values
    events_ids = events["result"].values
    events_dir = {}

    for event_id, (event_start, event_end) in zip(events_ids, events_dates):
        if event_id in events_dir:
            events_dir[event_id].append((event_start, event_end))
        else:
            events_dir[event_id] = [(event_start, event_end)]

    current = [i for i in cols if "current" in i]
    voltage = [i for i in cols if "voltage" in i]
    pressure = [i for i in cols if "pressure" in i]
    temperature = [i for i in cols if "temperature" in i]
    rates = [i for i in cols if "rate" in i]
    other = [i for i in cols if i not in [*current, *voltage, *pressure, *temperature, *rates]]

    groups = [current, voltage, pressure, temperature, rates, other]

    for group in groups:
        fig = plt.figure(figsize=(20, 15))

        for j, group_series in enumerate(group):
            

            data_to_plot = data[group_series]

            # if "time" in cols:
            #     data_to_plot["time"] = data["time"]

            ax = fig.add_subplot(len(group), 1, j+1)
            # ax.plot(data_to_plot.rolling(4, min_periods=1).mean())
            ax.plot(data_to_plot, alpha=0.75)

            # Adding events ranges into plots to analyze dependecies
            for i, (ev_id, dates_list) in enumerate(events_dir.items()):
                # mask = [(data_to_plot.index >= start_date) & (data_to_plot.index <= end_date) for start_date, end_date in dates_list]
                masks = pd.DataFrame()
                # print(f"Len of list {len(dates_list)}")
                for start_date, end_date in dates_list:
                    # if type(data_to_plot.index) == pd.DatetimeIndex:
                    msk = pd.Series((data_to_plot.index >= start_date) & (data_to_plot.index <= end_date))
                    # else:
                    #     msk = pd.Series((data_to_plot["time"] >= start_date) & (data_to_plot["time"] <= end_date))
                    # print(f"Sum of msk {np.sum(msk)}")
                    masks = pd.concat([masks, msk], axis=1)
                    # print(masks)
                
                # print(f"Shape {masks.shape}")
                masks.index = data_to_plot.index
                mask = masks.any(axis=1)
                # print(mask.shape)
                if mask.sum():
                    data_masked = data_to_plot.copy()
                    data_masked.loc[~mask] = np.nan
                    ax.plot(data_masked, plt.rcParams['axes.prop_cycle'].by_key()['color'][i%10], label=ev_id, linewidth=3)

            ax.set_title(f"{group_series} in Well {well_id}")
            ax.set_xlabel("Date")
            ax.set_ylabel(f"{group_series}")
            ax.grid()
            ax.legend()
            plt.tight_layout()
    


def feature_comparing(data_in: pd.DataFrame):
    data = data_in.copy()
    data = data.drop(["time", "well"], axis=1)
    cols = data.columns

    data_events = data[data["event_id"] != 0]
    data_clear = data[data["event_id"] == 0]

    # fig = plt.figure(figsize=(20, 15))
    for i, col in enumerate(cols):
        plt.sublot(3, len(cols)//3, i+1)
        sns.kdeplot()

