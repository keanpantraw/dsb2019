#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from functools import reduce
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import json
from functools import partial
from dsb2019.models.coeff import ThresholdClassifier

from dsb2019.models.tracking import track_experiment, track_submission_info
from dsb2019.data.validation import InstallationFold, cross_validate, quad_kappa
from dsb2019.visualization import session_browser
from dsb2019.data import DATA_DIR
from dsb2019.data import adv_validation
from dsb2019.models import MODELS_DIR
from dsb2019.models.nn import NN, make_nn_trainer
from dsb2019.models.lr_finder import LRFinder
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import hyperopt
from hyperopt import hp, fmin, Trials, tpe, STATUS_OK
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from pathlib import Path
import os

tqdm.pandas()
pd.options.display.max_rows=999


# In[ ]:


DATA_DIR = Path("/home/bfilippov/oss/dsb2019/dsb2019/data")
MODELS_DIR=Path("/home/bfilippov/oss/dsb2019/dsb2019/models")
train = pd.read_csv(DATA_DIR / 'raw/train.csv')
test = pd.read_csv(DATA_DIR / 'raw/test.csv')
train_labels = pd.read_csv(DATA_DIR / 'raw/train_labels.csv')
submission = pd.read_csv(DATA_DIR / 'raw/sample_submission.csv')


# In[ ]:


os.chdir("/home/bfilippov/oss/dsb2019/dsb2019/notebooks")


# In[ ]:


games = ['Scrub-A-Dub', 'All Star Sorting',
       'Air Show', 'Crystals Rule', 
       'Dino Drink', 'Bubble Bath', 'Dino Dive', 'Chow Time',
       'Pan Balance', 'Happy Camel',
       'Leaf Leader']
assessments = ['Mushroom Sorter (Assessment)',
       'Bird Measurer (Assessment)',
       'Cauldron Filler (Assessment)',
       'Cart Balancer (Assessment)', 'Chest Sorter (Assessment)']
worlds = ['NONE', 'MAGMAPEAK', 'TREETOPCITY', 'CRYSTALCAVES']

def unwrap_event_data(df):
    unwrapped=pd.DataFrame(data=list(df.event_data.apply(json.loads).values))
    unwrapped.drop("event_code", axis=1, inplace=True)
    return pd.concat([unwrapped.reset_index(),df.reset_index()],axis=1)


event_data_features = ["duration", "round", "level", "position", "total_duration", "weight", "misses"]


def extract_basic_stats(df):
    stats = ["min", "max", "mean", "std"]
    df = df[[f for f in event_data_features if f in df.columns]].reindex(columns=event_data_features)
    result = []
    for column, stats in df.agg(stats).to_dict().items():
        result.extend([(k+"_"+column, v) for k,v in stats.items()])
    return result


def extract_chow_time(df):
    cols = ["round", "event_id", "resources", "target_weight", "game_session"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.reindex(columns=cols)
    df = df[df["round"]>0]
    
    def calculate_features(df):
        data = df[df.event_id=="63f13dd7"]
        if len(data):
            data=data.iloc[0]
        else:
            return pd.Series({"optimum":None, "complete": 0})
        resources = data["resources"]
        target = data["target_weight"]
        optimum = 0
        cnt = 0
        for v in sorted(resources)[::-1]:
            if v+cnt>target:
                continue
            else:
                cnt+=v
                optimum+=1
                if cnt==target:
                    break
        n_turns = sum(df.event_id=="4ef8cdd3")
        complete = sum(df.event_id=="56817e2b")
        return pd.Series({"optimum":n_turns / optimum, "complete": complete})
    feature =  df.groupby(["game_session", "round"]).apply(calculate_features).mean()
    if len(feature)==0:
        return [("chowtime_optimum", None), ("chowtime_complete", None)]
    return [("chowtime_optimum", feature["optimum"]), ("chowtime_complete", feature["complete"])]


def extract_leaf_leader(df):
    cols = ["round", "event_id", "target_weight", "game_session"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.reindex(columns=cols)
    df = df[df["round"]>0]
    
    def calculate_features(df):
        data = df[df.event_id=="f32856e4"]
        if len(data):
            data=data.iloc[0]
        else:
            return pd.Series({"optimum": None, "complete": 0})
        target = data["target_weight"]
        optimum = 0
        cnt = 0
        for v in [4, 4, 2, 2, 2, 2, 1, 1]:
            if v+cnt>target:
                continue
            else:
                cnt+=v
                optimum+=1
                if cnt==target:
                    break
        n_turns = sum(df.event_id=="262136f4")
        complete = sum(df.event_id=="b012cd7f")
        return pd.Series({"optimum": n_turns / optimum, "complete": complete})
    feature =  df.groupby(["game_session", "round"]).apply(calculate_features).mean()
    if len(feature)==0:
        return [("leafleader_optimum", None), ("leafleader_complete", None)]
    return [("leafleader_optimum", feature["optimum"]), ("leafleader_complete", feature["complete"])]


def extract_happy_camel(df):
    cols = ["round", "event_id", "has_toy", "bowls", "game_session"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.reindex(columns=cols)
    df = df[df["round"]>0]
    
    def calculate_features(df):
        data = df[df.event_id=="c2baf0bd"]
        if len(data):
            data=data.iloc[0]
        else:
            return pd.Series({'optimum':None, 'n_toy_detected':None, "complete":0})
    
        optimum = len(data["bowls"])
        turns = df[df.event_id=="6bf9e3e1"]
        n_turns = len(turns)
        n_toy_detected = turns["has_toy"].sum()
        complete = sum(df.event_id=="36fa3ebe")
        return pd.Series({'optimum':(n_turns / optimum), 'n_toy_detected':n_toy_detected, "complete": complete})
    feature =  df.groupby(["game_session", "round"]).apply(calculate_features).mean()
    if len(feature)==0:
        return [("happycamel_optimum", None), ("happycamel_detections", None), 
                ("happycamel_complete", None)]

    return [("happycamel_optimum", feature["optimum"]), ("happycamel_detections", feature["n_toy_detected"]), 
            ("happycamel_complete", feature["complete"])]


def extract_pan_balance(df):
    cols = ["round", "event_id", "target_weight", "starting_weights", "game_session"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.reindex(columns=cols)
    df = df[df["round"]>0]
    
    def calculate_features(df):
        data = df[df.event_id=="a592d54e"]
        if len(data):
            data=data.iloc[0]
        else:
            return pd.Series({"optimum": None, "complete": 0})
        target = data["target_weight"]
        start = data["starting_weights"]
        optimum = max(abs(target - start), 1)
        n_turns = sum(df.event_id.isin(("e7561dd2", "804ee27f")))
        complete = sum(df.event_id=="1c178d24")
        return pd.Series({"optimum": n_turns / optimum, "complete": complete})
    feature =  df.groupby(["game_session", "round"]).apply(calculate_features).mean()
    if len(feature)==0:
        return [("panbalance_optimum", None), ("panbalance_complete", None)] 
    return [("panbalance_optimum", feature["optimum"]), ("panbalance_complete", feature["complete"])]


def extract_scrubadub(df):
    cols = ["round", "event_id", "game_session", "correct", "event_code", "animals"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.reindex(columns=cols)
    df = df[df["round"]>0]
    
    def calculate_features(df):
        data = df[df.event_id=="26fd2d99"]
        if len(data):
            data=data.iloc[0]
        else:
            return pd.Series({"performance":None, "precision": None, "complete": 0})
        n_animals = len(data["animals"])
        complete = sum(df.event_id=="08fd73f3")
        df = df[(df.event_id=="5c3d2b2f")&(df.event_code==4020)]
        return pd.Series({"performance": len(df) / n_animals, "precision": df["correct"].sum()/n_animals, "complete": complete})
    feature =  df.groupby(["game_session", "round"]).apply(calculate_features).mean()
    if len(feature)==0:
        return [("scrubadub_performance", None), ("scrubadub_precision", None), ("scrubadub_complete", None)]
    return [("scrubadub_performance", feature["performance"]), ("scrubadub_precision", feature["precision"]),
            ("scrubadub_complete", feature["complete"])]    


def extract_allstarsorting(df):
    cols = ["round", "event_id", "game_session", "correct", "event_code", "houses"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.reindex(columns=cols)
    df = df[df["round"]>0]
    
    def calculate_features(df):
        data = df[df.event_id=="2c4e6db0"]
        if len(data):
            data=data.iloc[0]
        else:
            return pd.Series({"performance":None, "precision": None, "complete": 0})
        n_animals = len(data["houses"])
        complete = sum(df.event_id=="ca11f653")
        df = df[df.event_id=="2dc29e21"]
        return pd.Series({"performance": len(df) / n_animals, "precision": df["correct"].sum()/n_animals, "complete": complete})
    feature =  df.groupby(["game_session", "round"]).apply(calculate_features).mean()
    if len(feature)==0:
        return [("allstarsorting_performance", None), ("allstarsorting_precision", None), ("allstarsorting_complete", None)]
    return [("allstarsorting_performance", feature["performance"]), ("allstarsorting_precision", feature["precision"]),
            ("allstarsorting_complete", feature["complete"])]  


def extract_dinodrink(df):
    cols = ["round", "event_id", "game_session", "correct", "event_code", "holes"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.reindex(columns=cols)
    df = df[df["round"]>0]
    
    def calculate_features(df):
        data = df[df.event_id=="f806dc10"]
        if len(data):
            data=data.iloc[0]
        else:
            return pd.Series({"performance": None, "precision": None, "complete": 0})
        n_animals = len(data["holes"])
        complete = sum(df.event_id=="16dffff1")
        df = df[df.event_id=="74e5f8a7"]
        return pd.Series({"performance": len(df) / n_animals, "precision": df["correct"].sum()/n_animals, "complete": complete})
    feature =  df.groupby(["game_session", "round"]).apply(calculate_features).mean()
    if len(feature)==0:
        return [("dinodrink_performance", None), ("dinodrink_precision", None), ("dinodrink_complete", None)]
    return [("dinodrink_performance", feature["performance"]), ("dinodrink_precision", feature["precision"]),
            ("dinodrink_complete", feature["complete"])]  


def extract_bubblebath(df):
    cols = ["round", "event_id", "game_session", "containers", "target_containers"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.reindex(columns=cols)
    df = df[df["round"]>0]
    
    def calculate_features(df):
        data = df[df.event_id=="1beb320a"]
        if len(data):
            data=data.iloc[0]
        else:
            return pd.Series({"performance": None, "complete": 0})
        target = data["target_containers"]
        complete = sum(df.event_id=="895865f3")
        df = df[df.event_id=="3bb91dda"]
        if len(df):
            return pd.Series({"performance": abs(target - df.iloc[0]["containers"]), "complete": complete})
        else:
            return pd.Series({"performance": None, "complete": 0})
    feature =  df.groupby(["game_session", "round"]).apply(calculate_features).mean()
    if isinstance(feature, pd.Series):
        return [("bubblebath_performance", None), ("bubblebath_complete", None)]
    return [("bubblebath_performance", feature["performance"]), ("bubblebath_complete", feature["complete"])] 


def extract_dinodive(df):
    cols = ["round", "event_id", "game_session", "correct", "target_water_level"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.reindex(columns=cols)
    df = df[df["round"]>0]
    
    def calculate_features(df):
        data = df[df.event_id=="7961e599"]
        if len(data):
            data=data.iloc[0]
        else:
            return pd.Series({"performance": None, "precision": None, "complete": 0})
        target = data["target_water_level"]
        dinos = [6, 6, 3, 3, 3, 3, 1, 1]
        opt=0
        n_animals=0
        for d in dinos:
            if opt+d>target:
                continue
            else:
                opt+=d
                n_animals+=1
                if opt==target:
                    break
        complete = sum(df.event_id=="00c73085")
        df = df[df.event_id=="c0415e5c"]
        return pd.Series({"performance": len(df) / n_animals, "precision": df["correct"].sum()/n_animals, "complete": complete})
    feature =  df.groupby(["game_session", "round"]).apply(calculate_features).mean()
    if len(feature)==0:
        return [("dinodive_performance", None), ("dinodive_precision", None), ("dinodive_complete", None)]
    return [("dinodive_performance", feature["performance"]), ("dinodive_precision", feature["precision"]),
            ("dinodive_complete", feature["complete"])]


def extract_airshow(df):
    cols = ["round", "event_id", "game_session", "correct"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.reindex(columns=cols)
    df = df[df["round"]>0]
    
    def calculate_features(df):
        complete =sum(df.event_id=="f5b8c21a")
        df = df[df.event_id=="28f975ea"]
        return pd.Series({"performance": len(df), "precision": df["correct"].sum(), "complete": complete})
    feature =  df.groupby(["game_session", "round"]).apply(calculate_features).mean()
    if len(feature)==0:
        return [("airshow_performance", None), ("airshow_precision", None), ("airshow_complete", None)]
    return [("airshow_performance", feature["performance"]), ("airshow_precision", feature["precision"]),
            ("airshow_complete", feature["complete"])]      


def extract_crystalsrule(df):
    cols = ["round", "event_id", "game_session", "correct"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.reindex(columns=cols)
    df = df[df["round"]>0]
    
    def calculate_features(df):
        complete = sum(df.event_id=="3323d7e9")
        df = df[df.event_id=="86c924c4"]
        return pd.Series({"performance": len(df), "precision": df["correct"].sum(), "complete": complete})
    feature =  df.groupby(["game_session", "round"]).apply(calculate_features).mean()
    if len(feature)==0:
        return [("crystalsrule_performance", None), ("crystalsrule_precision", None), ("crystalsrule_complete", None)]
    return [("crystalsrule_performance", feature["performance"]), ("crystalsrule_precision", feature["precision"]),
            ("crystalsrule_complete", feature["complete"])]          


game_funcs={
    "Chow Time": extract_chow_time,
    "Leaf Leader": extract_leaf_leader,
    "Happy Camel": extract_happy_camel,
    "Pan Balance": extract_pan_balance,
    "Scrub-A-Dub": extract_scrubadub,
    "All Star Sorting": extract_allstarsorting,
    "Dino Drink": extract_dinodrink,
    "Bubble Bath": extract_bubblebath,
    "Dino Dive": extract_dinodive,
    "Air Show": extract_airshow,
    "Crystals Rule": extract_crystalsrule,
}


def extract_game_stats(df, title):
    return game_funcs[title](df)


def make_counters(df, column):
    return {k: v for k, v in df.groupby(column)[column].count().to_dict().items()}

    
def process_log(df):
    assessment_title=df.title.iloc[-1]   
    world=df.world.iloc[-1]

    history = df.iloc[:-1]
    history = history[history.type.isin(["Game", "Assessment"])].copy()

    def calculate_ratios(df):
        n_correct=df.correct_move.sum()
        n_incorrect=df.wrong_move.sum()
        ratio=n_correct/(n_correct+n_incorrect)
        return n_correct, n_incorrect, ratio
    
    def make_move_stats(df, title,n_lags=2):
        df=df.copy()
        if len(df):
            df = unwrap_event_data(df)
        if "correct" in df.columns:
            df["correct_move"] = df.correct == True
            df["wrong_move"] = df.correct == False
        else:
            df["correct_move"]=False
            df["wrong_move"]=False
        result = []
        result.extend(zip([f"n_correct_{title}", f"n_incorrect_{title}", f"global_ratio_{title}"], calculate_ratios(df)))
        result.extend(extract_game_stats(df, title))
        #result.extend(extract_basic_stats(df))
        if n_lags:
            last_sessions = df.game_session.unique()[-n_lags:]
            for i in range(n_lags):
                if i < len(last_sessions): 
                    result.extend(zip([f"n_correct_{title}_{i}", f"n_incorrect_{title} {i}",f"ratio_{title}_{i}"], 
                                      calculate_ratios(df[df.game_session==last_sessions[i]])))
                else:
                    result.extend(zip([f"n_correct_{title}_{i}", f"n_incorrect_{title}_{i}",f"ratio_{title}_{i}"], [None, None, None]))
        return {k: v for k, v in result}
    
    
    result = {"title": assessments.index(assessment_title),
              "world": worlds.index(world),
              "n_activities": df[df.type=="Activity"].game_session.nunique(),
              "n_games": df[df.type=="Game"].game_session.nunique(),
              "event_code_count": df.event_code.nunique(),
              "event_id_count": df.event_id.nunique(),
              "title_count": df.title.nunique(),
              "n_actions": len(df),
              "world_title_count": df[df.world==world].title.nunique(),
             }
    
    def make_game_features(game):
        result = {}
        stats=history[history.title==game]
        stats_features=make_move_stats(stats, game)
        stats_features[f"{game}_event_code_count"] = stats.event_code.nunique()
        stats_features[f"{game}_event_id_count"] = stats.event_id.nunique()
        stats_features[f"{game}_session_id_count"] = stats.game_session.nunique()
        stats_features[f"{game}_n_actions"] = len(stats)
        result.update(stats_features)
        result.update({f"{game}_{k}": v for k, v in make_counters(stats, "event_id").items()})
        result.update({f"{game}_{k}": v for k, v in make_counters(stats, "event_code").items()})
        return result
    
    for f in Parallel(n_jobs=cpu_count())(delayed(make_game_features)(game) for game in games):
        result.update(f)
    #for game in games:
    #    result.update(make_game_features(game))
    world_games = history[history.world==world]
    
    def make_world_features(game):
        result = {}
        stats=world_games[world_games.title==game]
        stats_features=make_move_stats(stats, game)
        stats_features = {f"world_{k}": v for k, v in stats_features.items()}
        stats_features[f"world_{game}_event_code_count"] = stats.event_code.nunique()
        stats_features[f"world_{game}_event_id_count"] = stats.event_id.nunique()
        stats_features[f"world_{game}_session_id_count"] = stats.game_session.nunique()
        stats_features[f"world_{game}_n_actions"] = len(stats)
        result.update(stats_features)
        result.update({f"world_{game}_{k}": v for k, v in make_counters(stats, "event_id").items()})
        result.update({f"world_{game}_{k}": v for k, v in make_counters(stats, "event_code").items()})
        return result
    
    for f in Parallel(n_jobs=cpu_count())(delayed(make_world_features)(game) for game in games):
        result.update(f)
    #for game in games:
    #    result.update(make_world_features(game))
    make_history_counters = partial(make_counters, history)
    result.update(make_counters(history, "event_id"))
    result.update(make_counters(history, "event_code"))
    return result


# In[ ]:


def process_installations(train_labels, train, process_log):
    result = []
    train=train.sort_values("timestamp")
    installations = train.groupby("installation_id")
    for i, game_session, title, installation_id, accuracy_group in tqdm(train_labels[["game_session", "title", "installation_id", "accuracy_group"]].itertuples(), 
                                                              total=len(train_labels), position=0):
        player_log = installations.get_group(installation_id).reset_index()
        player_log = player_log.sort_values("timestamp")
        log_length = player_log[(player_log.game_session==game_session) & (player_log.title==title)].index[0]
        original_log = player_log
        player_log = player_log.iloc[:(log_length + 1)]
        #player_log["target_game_session"] = game_session
        features = process_log(player_log)
        features["installation_id"] = installation_id
        features["accuracy_group"] = accuracy_group
        result.append(features)
    return pd.DataFrame(data=result).fillna(-1)


# In[ ]:


pd.set_option('mode.chained_assignment', 'raise')
#train_features = process_installations(train_labels, train, process_log)
train_features = pd.read_csv(DATA_DIR/"processed/train_features.csv")


# In[ ]:


bad_features = ["session_id_count", "event_id_count", "mean_round", "n_actions", "n_activities"]


# In[ ]:


def get_duplicate_features(features, bad_features):
    to_remove = set([])
    counter = 0
    feature_names=[f for f in features.columns if f not in ("installation_id", "game_session", "accuracy_group")]
    for feat_a in tqdm(feature_names):
        for feat_b in feature_names:
            if feat_a != feat_b and feat_a not in to_remove and feat_b not in to_remove:
                c = np.corrcoef(features[feat_a], features[feat_b])[0][1]
                if c > 0.995:
                    counter += 1
                    to_remove.add(feat_b)
                    if feat_b in bad_features or feat_a in bad_features:
                        to_remove.add(feat_a)
                    #print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(counter, feat_a, feat_b, c))
    for bf in bad_features:
        to_remove.add(bf)
    print(f"{len(to_remove)} features were removed ({round(len(to_remove)/len(feature_names)*100, 2)}% of all features)")
    return list(to_remove)
    
#duplicate_features = get_duplicate_features(train_features, bad_features)

#useful_features = [f for f in train_features.columns if f not in duplicate_features]
#useful_features
useful_features = list(train_features.columns)


# In[ ]:


#train_features=train_features[useful_features].copy()


# In[ ]:


def lgb_quad_kappa(preds, true):
    true = true.get_label()
    #preds = preds.reshape((4, -1)).argmax(axis=0)
    preds = np.rint(preds)
    preds = np.maximum(0, preds)
    preds = np.minimum(3, preds)
    return "quad_kappa", quad_kappa(true, preds), True


def make_features(df):
    return df.drop(["installation_id", "accuracy_group"], axis=1), df.accuracy_group.values

def make_features_wrapper(train, test):
    def make_features(df):
        return df.drop(["installation_id", "accuracy_group"], axis=1), df.accuracy_group.values
    
    return make_features(train), make_features(test) 


def make_predictions(model,x_test_all,y_test):
    preds=model.predict(x_test_all)
    #preds = np.rint(preds)
    #preds = np.maximum(0, preds)
    #preds = np.minimum(3, preds)
    return preds,y_test


# In[ ]:


def process_test_installations(test):
    test = test.sort_values("timestamp")
    test=test.groupby("installation_id").progress_apply(process_log).reset_index()
    test.columns = ["installation_id", "features"]
    result = []
    for i, installation_id, feature in test.itertuples():
        result.append(feature)
        feature["installation_id"]=installation_id
    return pd.DataFrame(result).fillna(-1)
#test_features=process_test_installations(test)
test_features=pd.read_csv(DATA_DIR/"processed/test_features.csv")

for useful_feature in useful_features:
    if useful_feature not in test_features.columns:
        test_features[useful_feature]=-1
        print("Missing feature", useful_feature)

#test_features=test_features[[c for c in useful_features if c != "accuracy_group"]].copy()


# In[ ]:


get_ipython().system('ls ../dsb2019/models')


# In[ ]:


with open("../dsb2019/models/regression_baseline_ass_params.json", "r") as f:
    validator_params=json.load(f)
selected_features = [f for f in train_features.columns if f not in ("installation_id", "game_session", "accuracy_group")]
validator = adv_validation.AdversarialValidator(validator_params, train_features, test_features, selected_features=selected_features,test_size=0.5)
validator.fit()
print(validator.roc_auc())
validator.shap_important_features()


# In[ ]:


validator.lgb_important_features()


# In[ ]:


def make_dummy_features(train_features, test_features):
    dummy_features = []
    encoded_features = ["title", "world"]
    for i, title in enumerate(assessments):
        fname = f"title_{i}"
        for features in (train_features, test_features):
            features[fname] = features["title"]==i
        dummy_features.append(fname)
    for i, world in enumerate(worlds):
        fname = f"world_{i}"
        for features in (train_features, test_features):
            features[fname] = features["world"]==i
        dummy_features.append(fname)
    return dummy_features, encoded_features


# In[ ]:


dummy_features, encoded_features = make_dummy_features(train_features, test_features)
nn_features = [f for f in (useful_features + dummy_features) if f not in encoded_features]
train_features = train_features[nn_features]
test_features = test_features[nn_features]


# In[ ]:


input_size = len([f for f in nn_features if f not in ("accuracy_group", "installation_id")])
train_nn = make_nn_trainer(DATA_DIR / "nn_regression.w8")


# In[ ]:


subtrain_installations=pd.Series(train_features.installation_id.unique()).sample(frac=1., random_state=2019)
subtrain_features=train_features[train_features.installation_id.isin(subtrain_installations.values)].copy()
def check_hyperparams(params):
    print(params)
    if "max_depth" in params:
        params["max_depth"] = int(params["max_depth"])
    if "num_leaves" in params:
        params["num_leaves"] = int(params["num_leaves"])

    train_baseline_with_params = partial(train_nn, params=params)
    cv=InstallationFold(n_splits=3)
    predictions = cross_validate(subtrain_features, subtrain_features.accuracy_group, make_features_wrapper, train_baseline_with_params, make_predictions,
                                cv=cv)
    return {
        "loss": np.mean([mean_squared_error(true, pred) for pred, true in predictions]),
        "status": STATUS_OK,
        "params": params
    }


def tune(check_params, learning_rate, param_space, n_tries=25):        
    trials = Trials()
    param_space = param_space.copy()
    param_space["learning_rate"] = learning_rate
    fmin(check_params,
         param_space, tpe.suggest, n_tries, trials=trials)
    best_params = trials.best_trial["result"]["params"]
    return best_params


def find_learning_rate(best_params, train_features):
    lr_finder = LRFinder(min_lr=1e-4, max_lr=1)
    nn = NN("", **best_params)
    X, y = make_features(train_features)
    X = nn.scaler.fit_transform(X.values.astype(np.float64))
    
    nn.model.compile(loss='mse', optimizer='sgd')
    nn.model.fit(X, y, batch_size=128, callbacks=[lr_finder], epochs=2)


# In[ ]:


param_space = {
    "input_size": input_size,
    "dense_size": hp.choice("dense_size", [50, 100, 150, 200, 250, 300, 350, 400]),
    "dropout_prob": hp.uniform("dropout_prob", 1e-10, 1), 
    "n_layers": hp.choice("n_layers", [1, 2, 3, 4, 5, 6]),
}
#with open("../dsb2019/models/nn_params.json", "r") as f:
#    best_params=json.load(f)
best_params=tune(check_hyperparams, 0.055, param_space, n_tries=100)


# In[ ]:


print(best_params)


# What was selected on 100% of the data
# 
# ```
# {'feature_fraction': 0.53,
#  'lambda_l1': 0.922950554822482,
#  'lambda_l2': 0.835047934936944,
#  'learning_rate': 0.006,
#  'max_depth': 11,
#  'metric': 'rmse',
#  'n_estimators': 10000,
#  'num_leaves': 31,
#  'objective': 'rmse',
#  'random_state': 2019,
#  'subsample': 0.9500000000000001}
# 
# ```

# In[ ]:


find_learning_rate(best_params, train_features)


# In[ ]:


best_params["learning_rate"]=0.055


# In[ ]:


with open("../dsb2019/models/nn_params.json", "w") as f:
    json.dump(best_params, f)


# In[ ]:


baseline_model=train_nn(train_features.drop(["installation_id", "accuracy_group"], axis=1), train_features.accuracy_group.values, 
               params=best_params)


# In[ ]:


predictions = cross_validate(train_features, train_features.accuracy_group, make_features_wrapper, partial(train_nn, params=best_params), 
                             make_predictions)
print(np.mean([mean_squared_error(true, pred) for pred, true in predictions]), [mean_squared_error(true, pred) for pred, true in predictions])


# In[ ]:


baseline_model.save_model(MODELS_DIR/ "nn_regressor.model")


# In[ ]:


features, target = make_features(train_features)
prediction=baseline_model.predict(features)
clf = ThresholdClassifier()
clf.fit(prediction, target)


# In[ ]:


print(clf.coef_)


# In[ ]:




