import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.utils import shuffle


class InstallationFold(GroupKFold):
    def __init__(self, installation_ids=None):
        super().__init__(n_splits=10)
        self.installation_ids = installation_ids

    def split(self, X, y, installation_ids=None):
        if installation_ids is None:
            installation_ids = self.installation_ids
        orig_indices = np.arange(len(X))
        shuffled_indices, installation_ids = shuffle(orig_indices, installation_ids, random_state=2019)
        for train, test in super().split(shuffled_indices, shuffled_indices, installation_ids):
            yield shuffled_indices[train], shuffled_indices[test]