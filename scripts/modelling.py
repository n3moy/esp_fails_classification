import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from catboost import CatBoostClassifier

import optuna
from optuna.samplers import TPESampler

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score



def cross_val(data, y_true, params=None, model='rf'):
    tscv = KFold(n_splits=5, shuffle=True)
    mae = []

    for train_index, test_index in tscv.split(data):

        cv_train, cv_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = y_true.iloc[train_index], y_true.iloc[test_index]
        
        if model == 'lgb':

            cv_train_ds = lgb.Dataset(cv_train, y_train, silent=True)
            cv_test_ds = lgb.Dataset(cv_test, y_test, silent=True)

            if isinstance(params, dict):
                booster = lgb.train(params, train_set=cv_train_ds, valid_sets=cv_test_ds, num_boost_round=params['num_boost_round'], early_stopping_rounds=50, verbose_eval=1000)
            else:
                booster = lgb.train({'objective':'classification'}, train_set=cv_train_ds, verbose_eval=1000)
            train_preds = booster.predict(cv_train)
            val_preds = booster.predict(cv_test)

        if model == 'catboost':

            if isinstance(params, dict):
                ct = CatBoostClassifier(verbose=False, eval_metric='MAE', **params)
            else:
                ct = CatBoostClassifier(verbose=False, eval_metric='MAE')
            ct.fit(cv_train, y_train)
            train_preds = ct.predict(cv_train)
            val_preds = ct.predict(cv_test)

        if model == 'rf':

            rf = RandomForestClassifier(random_state=42)
            rf.fit(cv_train, y_train)
            train_preds = rf.predict(cv_train)
            val_preds = rf.predict(cv_test)

        train_mae = metrics.roc_auc_score(y_train, train_preds)
        val_mae = metrics.roc_auc_score(y_test, val_preds)

        print('MAE on train set: {}'.format(train_mae))
        print('MAE on valid set: {}'.format(val_mae))

        mae.append(val_mae)

    return np.mean(mae)


def objective(trial, X_train, y_train, model: str) -> float:

    if model == "lgb":
        params = {
            'objective': 'regression',
            'boosting': 'gbdt',
            'metric': 'rmse',
            # 'device': 'gpu',
            'verbosity': 1000,
            # 'max_depth': trial.suggest_int('max_depth', 4, 15),
            'max_bin': trial.suggest_int('max_bin', 100, 400),
            'num_boost_round': 5000,
            'bagging_fraction': trial.suggest_loguniform('bagging_fraction', 0.1, 0.9),
            'num_leaves': trial.suggest_int('num_leaves', 150, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.0001, 0.3),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200)
        }

    if model == "catboost":
        params = {
            'objective': 'regression',
            'boosting': 'gbdt',
            'metric': 'rmse',
            # 'device': 'gpu',
            'verbosity': 1000,
            # 'max_depth': trial.suggest_int('max_depth', 4, 15),
            'max_bin': trial.suggest_int('max_bin', 100, 400),
            'num_boost_round': 5000,
            'bagging_fraction': trial.suggest_loguniform('bagging_fraction', 0.1, 0.9),
            'num_leaves': trial.suggest_int('num_leaves', 150, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.0001, 0.3),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200)
        }

    val_mae = cross_val(X_train, y_train, params, model=model)
    print('Validation MAE: {}'.format(val_mae))

    return val_mae


def train_model(algo: str, X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series):
    """
    This function is designed to
    
    """


def optimize_cat():
    pass


#TODO Ебнуть кластеризацию по параметрам и посмотреть разницу по поломкам

def clusterization():
    pass



def evaluate_model(alg, train, target, predictors,  early_stopping_rounds=1):
    
   
    #Fit the algorithm on the data
    alg.fit(train[predictors], target['FAILURE_TARGET'], eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(train[predictors])
    dtrain_predprob = alg.predict_proba(train[predictors])[:,1]
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False) 
    feat_imp.plot(kind='bar', title='Feature Importance', color='g') 
    plt.ylabel('Feature Importance Score')
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(target['FAILURE_TARGET'].values, dtrain_predictions))
    print("AUC Score (Balanced): %f" % metrics.roc_auc_score(target['FAILURE_TARGET'], dtrain_predprob))


def evaluate_results(y_true, y_preds):
    ms = {
        "Precision": metrics.precision_score, "Recall": metrics.recall_score, "ROC_AUC": metrics.roc_auc_score
    }

    for metric in ms.items():
        name = metric[0]
        m = metric[1]
        print(f"{name} --- {m(y_true, y_preds)}")


def evaluate_results2(data_in: pd.DataFrame, forecast_window: int=14) -> pd.DataFrame:
    data = data_in.copy()
    cols = data.columns

    if "time" not in cols:
        raise KeyError("'Time' column is not found in data")

    data["time"] = pd.to_datetime(data["time"])
    data = data.sort_values(by="time", ascending=True)
    data.loc[(data["event_id"] != 0), "failure_date"] = data.loc[(data["event_id"] != 0), "time"]
    data["failure_date"] = data["failure_date"].bfill().fillna(data["time"].max())
    data["failure_date"] = pd.to_datetime(data["failure_date"])

    data["Y_FAIL_sumxx"] = 0
    data["Y_FAIL_sumxx"] = (data["predicted"].rolling(min_periods=1, window=(forecast_window)).sum())

    # if a signal has occured in the last 14 days, the signal is 0.
    data['Y_FAILZ'] = np.where((data.Y_FAIL_sumxx > 1), 0, data.predicted)
    #sort the data by id and date.
    data = data.sort_values(by=["time"], ascending=[True])

    #create signal id with the cumsum function.
    data['SIGNAL_ID'] = data['Y_FAILZ'].cumsum()
    df_signals = data[data['Y_FAILZ'] == 1].copy()
    df_signal_date = df_signals[['SIGNAL_ID','time']].copy()
    df_signal_date = df_signal_date.rename(index=str, columns={"time": "SIGNAL_DATE"})
    data = data.merge(df_signal_date, on=['SIGNAL_ID'], how='outer')

    data['C'] = data['failure_date'] - data['SIGNAL_DATE']
    data['WARNING'] = data['C'] / np.timedelta64(1, 'D')

    data["true_failure"] = data["event_id"] != 0
    data["FACT_FAIL_sumxx"] = 0
    data["FACT_FAIL_sumxx"] = (data["true_failure"].rolling(min_periods=1, window=forecast_window).sum())

    # if a signal has occured in the last 90 days, the signal is 0.
    data['actual_failure'] = np.where((data.FACT_FAIL_sumxx>1), 0, data.true_failure)

    # define a true positive
    data['TRUE_POSITIVE'] = np.where(((data.actual_failure == 1) & (data.WARNING<=forecast_window) & (data.WARNING>=0)), 1, 0)
    # define a false negative
    data['FALSE_NEGATIVE'] = np.where((data.TRUE_POSITIVE==0) & (data.actual_failure==1), 1, 0)
    # define a false positive
    data['BAD_S'] = np.where((data.WARNING<0) | (data.WARNING>=forecast_window), 1, 0)
    data['FALSE_POSITIVE'] = np.where(((data.Y_FAILZ == 1) & (data.BAD_S==1)), 1, 0)
    data['bootie']=1
    data['CATEGORY'] = np.where((data.FALSE_POSITIVE==1),'FALSE_POSITIVE',
                                        (np.where((data.FALSE_NEGATIVE==1),'FALSE_NEGATIVE',
                                                    (np.where((data.TRUE_POSITIVE==1),'TRUE_POSITIVE','TRUE_NEGATIVE')))))
    table = pd.pivot_table(data, values=['bootie'], columns=['CATEGORY'], aggfunc=np.sum)

    return table, data


def plot_signals(data_list: list):
    data_wells = data_list.copy()
    plt.figure(figsize=(30, 20))

    for i, data in enumerate(data_wells):

        signal_dates = data[data['TRUE_POSITIVE'] == 1]["time"].values
        failure_dates = data[data["actual_failure"] == 1]["time"].values

        plt.subplot(2, 2, i+1)
        plt.plot(data["time"], data["current"], label="Current")

        for signal_date in signal_dates:
            plt.axvline(signal_date, 0 , data["current"].max(), c="g", label="SIGNAL")

        for failure_date in failure_dates:
            plt.axvline(failure_date, 0 , data["current"].max(), c="r", label="FAILURE", alpha=0.75)

        # plt.legend()
        plt.title(f"Current in Well #{i+1}")
        plt.xlabel("DATE")
        plt.ylabel("CURRENT, A")

    plt.show()

