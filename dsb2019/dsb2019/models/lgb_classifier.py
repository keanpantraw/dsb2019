import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from functools import reduce
from functools import partial

from dsb2019.models.tracking import track_experiment, track_submission_info
from dsb2019.data.validation import InstallationFold, cross_validate, quad_kappa


def lgb_quad_kappa(preds, true):
    true = true.get_label()
    preds = preds.reshape((4, -1)).argmax(axis=0)
    return "quad_kappa", quad_kappa(true, preds), True
    
    
def train_baseline(x_train,y_train, params=None):
    x_train_all, x_val_all,y_train_all,y_val_all = train_test_split(
        x_train,y_train,
        test_size=0.15,
        random_state=2019,
    )
    train_set = lgb.Dataset(x_train_all, y_train_all)
    val_set = lgb.Dataset(x_val_all, y_val_all)

    return lgb.train(params, train_set, num_boost_round=10000, early_stopping_rounds=2000, valid_sets=[train_set, val_set], verbose_eval=100,
                    feval=lgb_quad_kappa)


def make_features_wrapper(*dataframes):
    def make_features(df):
        return df.drop(["installation_id", "accuracy_group", "target_game_session"], axis=1), df.accuracy_group.values
    
    result = tuple([make_features(df) for df in dataframes]) 
    if len(result) == 1:
        return result[0]
    return result


def make_predictions(model,x_test_all,y_test):
    pred=model.predict(x_test_all).argmax(axis=1)
    return pred,y_test


def make_submission(test_features, model):
    installations = test_features.installation_id.values
    test, _ = make_features_wrapper(test_features)
    predictions, _ = make_predictions(model, test, None)
    return pd.DataFrame(data={"installation_id": installations, "accuracy_group": predictions})
