import numpy as np


class GenericPredictor:
    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, features):
        return self.predictor.predict(features)

    @classmethod
    def preprocess_features(cls, features, useful_features=None, **unused_context):
        return features[useful_features]
    

class Ensemble:
    def __init__(self, predictor_list):
        self.predictor_list = predictor_list

    def predict(self, features, **context):
        predictions = []
        for weight, predictor in self.predictor_list:
            processed_features = predictor.preprocess_features(features, **context)
            processed_features = processed_features.drop([f for f in ["installation_id", "accuracy_group"] if f in processed_features.columns], axis=1)
            predictions.append(predictor.predict(processed_features) * weight)
        return np.sum(predictions, axis=0)


def load_nn(path):
    nn = NN.load(path)
    return nn

def load_lgb(path):
    model = lgb.Booster(model_file=path)
    return GenericPredictor(model)
