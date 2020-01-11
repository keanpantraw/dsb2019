from typing import NamedTuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap


class AdvData(NamedTuple):
    x_train: pd.DataFrame
    y_train: np.array
    x_test: pd.DataFrame
    y_test: np.array


class AdversarialValidator:
    def __init__(self, parameters: dict, train: pd.DataFrame, test: pd.DataFrame, test_size=0.2, selected_features=None):
        self.parameters = parameters.copy()
        self.parameters["objective"] = "binary"
        self.parameters["metric"] = "auc"
        if selected_features is not None:
            train = train[selected_features]
            test = test[selected_features]
        self.data_ = self.prepare_data(train, test, test_size)
    
    def clear(self):
        del self.model_
        del self.data_

    def prepare_data(self, train: pd.DataFrame, test: pd.DataFrame, test_size=0.5):
        X = pd.concat([train, test], axis=0, ignore_index=True)
        y = np.asarray(([0] * len(train)) + ([1] * len(test)))
        x_train_all, x_val_all,y_train_all,y_val_all = train_test_split(
            X, y,
            test_size=test_size,
            random_state=2019,
        )
        self.data_ = AdvData(x_train_all, y_train_all, x_val_all, y_val_all)
        return self.data_

    def fit(self):
        x_train_all, x_val_all,y_train_all,y_val_all = train_test_split(
            self.data_.x_train, self.data_.y_train,
            test_size=0.15,
            random_state=2019,
        )
        train_set = lgb.Dataset(x_train_all, y_train_all)
        val_set = lgb.Dataset(x_val_all, y_val_all)
        self.model_ = lgb.train(self.parameters, train_set, num_boost_round=10000, early_stopping_rounds=2000, valid_sets=[val_set], verbose_eval=100)
        return self

    def roc_auc(self) -> float:
        return roc_auc_score(self.data_.y_test, self.model_.predict(self.data_.x_test))
    
    def lgb_important_features(self):
        return lgb.plot_importance(self.model_, max_num_features=20)

    def shap_important_features(self):
        shap_values = shap.TreeExplainer(self.model_).shap_values(self.data_.x_test)
        return shap.summary_plot(shap_values, self.data_.x_test, plot_type="bar")
