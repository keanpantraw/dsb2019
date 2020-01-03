from unittest import TestCase
import numpy as np
from dsb2019.models.coeff import ThresholdClassifier
from sklearn.utils.estimator_checks import check_estimator


class ThresholdClassifierTestCase(TestCase):
    def test_estimator_contract(self):
        size = 10_000
        X = np.random.uniform(size=size)
        y = np.random.randint(4, size=size)
        clf = ThresholdClassifier()
        clf.fit(X, y)

        X1 = np.random.uniform(size=size)

        pred = clf.predict(X1)

        self.assertEqual(len(pred), len(X1))
        self.assertTrue(np.max(pred)<=3)
        self.assertTrue(np.min(pred)>=0)
