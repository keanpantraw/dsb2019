import contextlib
import os
import joblib
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from functools import reduce, partial
import json

from tqdm import tqdm
from dsb2019.features.game import games


def unwrap_event_data(df):
    unwrapped=pd.DataFrame(data=list(df.event_data.apply(json.loads).values))
    return pd.concat([unwrapped.reset_index(),df.reset_index()],axis=1)


def process_test_installations(test, process_log):
    test_labels = test[["game_session", "title", "installation_id"]].drop_duplicates()
    test_labels["accuracy_group"] = None
    return process_installations(test_labels, test, process_log)


def process_installations(train_labels, train, process_log, n_installations_in_chunk=100, n_jobs=None):
    installation_ids = train.installation_id.unique()
    chunk_size = n_installations_in_chunk
    n_jobs = n_jobs if n_jobs is not None else cpu_count()
    tasks = []
    for p, i_low in enumerate(tqdm(range(0, len(installation_ids), chunk_size), desc="Generating tasks", position=0)):
        i_high = min(len(installation_ids), i_low + chunk_size)
        installation_ids_chunk = installation_ids[i_low:i_high]
        train_labels_chunk = train_labels[train_labels.installation_id.isin(installation_ids_chunk)].copy()
        train_chunk = train[train.installation_id.isin(installation_ids_chunk)].copy()
        task = joblib.delayed(process_installations_single)(train_labels_chunk, train_chunk, process_log, position=p+1)
        tasks.append(task)
    
    result = []
    with tqdm_joblib(tqdm(desc="Completing tasks", total=len(tasks), position=0)) as progress_bar:
        with joblib.Parallel(n_jobs=n_jobs) as workers:
            for result_df in workers(tasks):
                result.append(result_df)
    return pd.concat(result, ignore_index=True)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback:
        def __init__(self, time, index, parallel):
            self.index = index
            self.parallel = parallel

        def __call__(self, index):
            tqdm_object.update()
            if self.parallel._original_iterator is not None:
                self.parallel.dispatch_next()
    
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def process_installations_single(train_labels, train, process_log, position=1):
    result = []
    train = train.sort_values("timestamp")
    installations = train.groupby("installation_id")
    for i, game_session, title, installation_id, accuracy_group in tqdm(train_labels[["game_session", "title", "installation_id", "accuracy_group"]].itertuples(), 
                                                              total=len(train_labels), position=position, desc=f"Processing chunk {position}"):
        player_log = installations.get_group(installation_id).reset_index()
        log_length = player_log[(player_log.game_session==game_session) & (player_log.title==title)].index[0]
        player_log = player_log.iloc[:(log_length + 1)]
        player_log["accuracy_group"] = accuracy_group
        player_log["target_game_session"] = game_session
        features = process_log(player_log)
        features["installation_id"] = installation_id
        features["target_game_session"] = game_session
        features["accuracy_group"] = accuracy_group
        result.append(features)
    df = pd.DataFrame(data=result)
    return df[sorted(df.columns)].fillna(-1)


class LogProcessor:
    def __init__(self, global_features, game_features):
        self.global_features = global_features
        self.game_features = game_features

    def __call__(self, df):
        history = df.iloc[:-1]
        history = history[history.type.isin(["Game", "Assessment"])].copy()
        
        result = {}
        for func in self.global_features:
            result.update(func(df))
        for game in games:
            game_feature_funcs = self.game_features.get(game, [])
            if game_feature_funcs:
                game_info=history[history.title==game].copy()
                if len(game_info):
                    game_info = unwrap_event_data(game_info)
                for func in game_feature_funcs:
                    result.update(func(game_info))
        return result
