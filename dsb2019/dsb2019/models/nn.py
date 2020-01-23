import tensorflow as tf
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


class NN:
    def __init__(self, **params):
        self.params = params
        self.learning_rate = params["learning_rate"]
        self.file_path = str(params["file_path"])
        self.scaler = params["scaler"]
        dense_size = params["dense_size"]
        n_layers = params["n_layers"]
        dropout_prob = params["dropout_prob"]
        input_size = params["input_size"]
        
        dense_sizes = np.linspace(dense_size, 1, num=n_layers+1, dtype=int)[:-1]
        
        
        def layer(size, dropout_prob):
            return [tf.keras.layers.Dense(dense_size, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_prob)]

        layers = [tf.keras.layers.Input(shape=(input_size,))]
        for dense_size in dense_sizes:
            layers.extend(layer(dense_size, dropout_prob))
        layers.append(tf.keras.layers.Dense(1, activation='relu'))
        self.model = tf.keras.models.Sequential(layers)

    def fit(self, x_train, y_train, epochs=10_000, patience=500): #yo shall be used just like lgb guess ya?
        x_train_all, x_val_all,y_train_all,y_val_all = train_test_split(
            x_train,y_train,
            test_size=0.15,
            random_state=2019,
        )
        self.scaler.fit(x_train_all.values)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        
        save_best = tf.keras.callbacks.ModelCheckpoint(self.file_path, save_weights_only=True, save_best_only=True, verbose=0)
        early_stop = tf.keras.callbacks.EarlyStopping(patience=patience)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                     patience=250, min_lr=0.0001)

        base_learning_rate = self.learning_rate
        divisor = np.linspace(2, 8, num=epochs)
        period = 500
        _2pi = np.pi * 2
        all_epochs = np.arange(0, epochs)
        schedule = np.sin(_2pi * all_epochs / period) * base_learning_rate/divisor + base_learning_rate

        #plt.plot(all_epochs, schedule)

        def get_learning_rate(epoch):
            return schedule[epoch]


        scheduler = tf.keras.callbacks.LearningRateScheduler(get_learning_rate)
        class Metrics(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if epoch % 50 == 0:
                    print(f"Epoch {epoch} loss={logs['loss']} val_loss={logs['val_loss']}")

        reporter=Metrics()
        self.model.fit(self.scaler.transform(x_train_all.astype(np.float64)), 
                y_train_all, 
                validation_data=(self.scaler.transform(x_val_all.astype(np.float64)), y_val_all),
                epochs=epochs,
                 callbacks=[save_best, early_stop, reporter, 
                 #scheduler,
                 reduce_lr
                 ], verbose=0, batch_size=x_train_all.shape[0])
        self.model.load_weights(self.file_path)

    def load_weights(self):
        self.model.load_weights(str(self.file_path))

    @staticmethod
    def load(model_path):
        weights_path = str(model_path) + "_weights"
        with open(model_path, "rb") as f:
            params = pickle.load(f)
            params["file_path"] = weights_path
        model = NN(**params)    
        model.load_weights()
        return model

    def save_model(self, model_path):
        weights_name = Path(model_path).name + "_weights"
        model_dir = Path(model_path).parent
        self.model.save_weights(str(model_dir / weights_name))
        with open(str(model_path), "wb") as f:
            pickle.dump(self.params, f)
        
    def predict(self, features):
        x = self.scaler.transform(features.values.astype(np.float64))
        return self.model.predict(x).flatten()

    @classmethod
    def make_dummy_features(cls, *feature_list, assessments, worlds):
        dummy_features = []
        encoded_features = ["title", "world"]
        for i, title in enumerate(assessments):
            fname = f"title_{i}"
            for features in feature_list:
                features[fname] = features["title"]==i
            dummy_features.append(fname)
        for i, world in enumerate(worlds):
            fname = f"world_{i}"
            for features in feature_list:
                features[fname] = features["world"]==i
            dummy_features.append(fname)
        return dummy_features, encoded_features
    
    @classmethod
    def preprocess_features(cls, features, useful_features=None, assessments=None, worlds=None, **unused_context):
        features = features.copy()
        dummy_features, encoded_features = cls.make_dummy_features(features, assessments=assessments, worlds=worlds)
        nn_features = [f for f in (useful_features + dummy_features) if f not in encoded_features and f !="accuracy_group" and f in features.columns]
        features = features[nn_features]
        return features


def make_nn_trainer(file_path, useful_features, assessments, worlds):
    def train_nn(x_train, y_train, params=None):
        params = params.copy()
        params["file_path"] = file_path
        params["scaler"] = StandardScaler()
        model = NN(**params)
        x_train = model.preprocess_features(x_train, useful_features=useful_features, assessments=assessments, worlds=worlds)
        x_train = x_train.drop([f for f in ["installation_id", "accuracy_group"] if f in x_train.columns], axis=1)
        model.fit(x_train, y_train)
        return model
    return train_nn
