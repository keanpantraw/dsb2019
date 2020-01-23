import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.utils import shuffle
from typing import NamedTuple
from functools import partial
from sklearn.metrics import cohen_kappa_score


class Predict(NamedTuple):
    true: np.array
    pred: np.array


class InstallationFold(GroupKFold):
    def __init__(self, n_splits=5, installation_ids=None):
        super().__init__(n_splits=n_splits)
        self.installation_ids = installation_ids

    def split(self, X, y, installation_ids=None):
        if installation_ids is None:
            installation_ids = self.installation_ids
        orig_indices = np.arange(len(X))
        shuffled_indices, installation_ids = shuffle(orig_indices, installation_ids, random_state=2019)
        for train, test in super().split(shuffled_indices, shuffled_indices, installation_ids):
            yield shuffled_indices[train], shuffled_indices[test]


def fit_fold(df, train_ix, test_ix, make_features, train_model, make_predictions):
    train = df.iloc[train_ix].copy()
    test = df.iloc[test_ix].copy()
    train_features, test_features = make_features(train, test)
    model = train_model(*train_features)
    test_pred, test_true = make_predictions(model, *test_features)
    return Predict(test_true, test_pred), model


def cross_validate(train, labels, make_features, train_model, make_predictions, cv=None):
    predicts = []
    models = []
    np.random.seed(2019)
    cv = InstallationFold() if cv is None else cv
    for ix_train, ix_test in cv.split(train, labels, train.installation_id.values):
        predict, model = fit_fold(train, ix_train, ix_test, make_features, train_model, make_predictions)
        predicts.append(predict)
        models.append(model)
    return predicts, models


quad_kappa = partial(cohen_kappa_score, weights="quadratic")