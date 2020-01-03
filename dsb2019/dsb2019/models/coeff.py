import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from functools import partial
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from dsb2019.data.validation import quad_kappa


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_iter=1000, random_state=2019):
        self.n_iter=n_iter
        self.random_state=random_state

    def _run_trial(self, X, y, params):
        threshold1 = params["threshold1"]
        threshold2 = threshold1 + abs(params["threshold2_delta"])
        threshold3 = threshold2 + abs(params["threshold3_delta"]) 
        pred = pd.cut(X, [-np.inf, threshold1, threshold2, threshold3, np.inf], labels = [0, 1, 2, 3])
        return {
           "loss": -quad_kappa(y, pred),
           "status": STATUS_OK,
           "coef": [threshold1, threshold2, threshold3]
        }

    def fit(self, X, y):
        class1_percentile = sum(y<1) / len(y) * 100
        class2_percentile = sum(y<2) / len(y) * 100
        class3_percentile = sum(y<3) / len(y) * 100
        threshold1_prior = np.percentile(X, class1_percentile)
        threshold2_prior = np.percentile(X, class2_percentile)
        threshold3_prior = np.percentile(X, class3_percentile)
        threshold2_delta_prior = threshold2_prior - threshold1_prior
        threshold3_delta_prior = threshold3_prior - threshold2_prior
        prior_std = (np.percentile(X, 99) - np.percentile(X, 1)) / 3
        space = {
            "threshold1": hp.normal("threshold1", threshold1_prior, prior_std),
            "threshold2_delta": hp.normal("threshold2_delta", threshold2_delta_prior, prior_std),
            "threshold3_delta": hp.normal("threshold3_delta", threshold3_delta_prior, prior_std)
        }

        partial_run = partial(self._run_trial, X, y)

        trials = Trials()
        fmin(partial_run, space=space,
             algo=tpe.suggest,
             max_evals=self.n_iter, rstate=np.random.RandomState(self.random_state), trials=trials)
        
        self.coef_ = trials.best_trial["result"]["coef"]
        return self

    def predict(self, X):
        return pd.cut(X, [-np.inf] + self.coef_ + [np.inf], labels = [0, 1, 2, 3])
