#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import os
from target_encoding import TargetEncoderClassifier, TargetEncoder
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
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import hyperopt
from hyperopt import hp, fmin, Trials, tpe, STATUS_OK
from multiprocessing import cpu_count
from joblib import Parallel, delayed

tqdm.pandas()
pd.options.display.max_rows=999


# In[3]:


train = pd.read_csv(DATA_DIR / 'raw/train.csv')
test = pd.read_csv(DATA_DIR / 'raw/test.csv')
train_labels = pd.read_csv(DATA_DIR / 'raw/train_labels.csv')
submission = pd.read_csv(DATA_DIR / 'raw/sample_submission.csv')


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

game_events={'Air Show': ['9b4001e4',
  '7423acbc',
  '58a0de5c',
  'e04fb33d',
  'a1bbe385',
  'f28c589a',
  'dcb1663e',
  '65abac75',
  '15ba1109',
  'd88ca108',
  'bcceccc6',
  '6f4bd64e',
  'f5b8c21a',
  '1575e76c',
  'd2659ab4',
  '14de4c5d',
  '28f975ea',
  'dcb55a27',
  '06372577'],
 'All Star Sorting': ['daac11b0',
  'c277e121',
  '1f19558b',
  'd45ed6a1',
  '9e4c8c7b',
  '363d3849',
  '26a5a3dd',
  '4b5efe37',
  'b7dc8128',
  '587b5989',
  '6043a2b4',
  'b1d5101d',
  'ca11f653',
  '2c4e6db0',
  'd02b7a8e',
  '2dc29e21',
  'b120f2ac',
  '1cc7cfca'],
 'Bubble Bath': ['6f4adc4b',
  '1cf54632',
  '90ea0bac',
  '55115cbd',
  '5859dfb6',
  '15eb4a7d',
  '0413e89d',
  '29a42aea',
  '8d84fa81',
  '99abe2bb',
  '99ea62f3',
  'a0faea5d',
  '6aeafed4',
  'ecc36b7f',
  '895865f3',
  '1beb320a',
  '3bb91dda',
  '85de926c',
  '8f094001',
  '857f21c0',
  'ad148f58',
  '1340b8d7',
  'c54cf6c5',
  'd06f75b5'],
 'Chow Time': ['47026d5f',
  '7d093bf9',
  '0330ab6a',
  'cb6010f8',
  '2230fab4',
  '0d1da71f',
  '7ec0c298',
  '6f445b57',
  'f93fc684',
  '19967db1',
  '7372e1a5',
  '9e6b7fb5',
  '56817e2b',
  '63f13dd7',
  'd185d3ea',
  '4ef8cdd3',
  'cfbd47c8'],
 'Crystals Rule': ['3ddc79c3',
  '48349b14',
  '44cb4907',
  'e720d930',
  '8b757ab8',
  '3babcb9b',
  '5154fc30',
  'cc5087a3',
  '5e3ea25a',
  '93edfe2e',
  '3323d7e9',
  '7cf1bc53',
  '17ca3959',
  'a1192f43',
  '86c924c4'],
 'Dino Dive': ['e3ff61fb',
  '29bdd9ba',
  '9de5e594',
  '709b1251',
  '28a4eb9a',
  'ab3136ba',
  '832735e1',
  '119b5b02',
  '87d743c1',
  '6088b756',
  '76babcde',
  'd3640339',
  '00c73085',
  '7961e599',
  'c0415e5c',
  '7d5c30a2'],
 'Dino Drink': ['77ead60d',
  '51311d7a',
  'e5734469',
  '4d911100',
  '89aace00',
  '7f0836bf',
  'a29c5338',
  'ab4ec3a4',
  '5be391b5',
  '4d6737eb',
  '9ed8f6da',
  '6c517a88',
  '6f8106d9',
  '16dffff1',
  'f806dc10',
  '792530f8',
  '74e5f8a7',
  '1996c610',
  'c6971acf'],
 'Happy Camel': ['c7fe2a55',
  'd9c005dd',
  '1af8be29',
  'a8a78786',
  '3bf1cf26',
  '69fdac0a',
  '8d7e386c',
  '0ce40006',
  'abc5811c',
  'd51b1749',
  'c189aaf2',
  'a7640a16',
  '05ad839b',
  '46b50ba8',
  '36fa3ebe',
  'c2baf0bd',
  'a2df0760',
  '3d8c61b0',
  '8af75982',
  '6bf9e3e1',
  '37db1c2f',
  '3bb91ced'],
 'Leaf Leader': ['3afde5dd',
  '8ac7cce4',
  '763fc34e',
  'e5c9df6f',
  'e57dd7af',
  '2a512369',
  '33505eae',
  '01ca3a3c',
  'fd20ea40',
  '86ba578b',
  '53c6e11a',
  '7dfe6d8a',
  '67aa2ada',
  '3b2048ee',
  'b012cd7f',
  'f32856e4',
  '262136f4',
  '86ba578b',
  '29f54413'],
 'Pan Balance': ['250513af',
  '9c5ef70c',
  '907a054b',
  'cf7638f3',
  'c51d8688',
  '15f99afc',
  '6cf7d25c',
  'e4d32835',
  '0086365d',
  'e080a381',
  'f3cd5473',
  '1c178d24',
  'a592d54e',
  '4074bac2',
  'bc8f2793',
  '804ee27f',
  'a5e9da97',
  '2a444e03',
  'e7561dd2'],
 'Scrub-A-Dub': ['73757a5e',
  '6d90d394',
  'd88e8f25',
  '2b9272f4',
  'ac92046e',
  'f7e47413',
  'f71c4741',
  '5dc079d8',
  '7040c096',
  '5a848010',
  '92687c59',
  '4a09ace1',
  'cf82af56',
  '08fd73f3',
  '26fd2d99',
  '5c3d2b2f',
  '37c53127',
  'dcaede90',
  'c1cac9a2']}

final_events=[
    ['47026d5f',
 '3afde5dd',
 '3ddc79c3',
 '6f4adc4b',
 '77ead60d',
 '9b4001e4',
 '250513af',
 'c7fe2a55',
 '73757a5e',
 'e3ff61fb',
 'daac11b0'],
    ['1cf54632',
 '7d093bf9',
 '8ac7cce4',
 '51311d7a',
 '29bdd9ba',
 'd9c005dd',
 '48349b14',
 '6d90d394',
 '9c5ef70c'],
['d88e8f25',
 'c277e121',
 '7423acbc',
 '1af8be29',
 '763fc34e',
 '90ea0bac',
 'e5734469',
 '9de5e594',
 '907a054b',
 '44cb4907',
 '0330ab6a'],
    ['2b9272f4',
 'e720d930',
 '709b1251',
 '4d911100',
 'cf7638f3',
 'e5c9df6f',
 'a8a78786',
 '55115cbd',
 'cb6010f8',
 '1f19558b',
 '58a0de5c'],
['5859dfb6',
 'e04fb33d',
 '28a4eb9a',
 'e57dd7af',
 '2230fab4',
 'c51d8688',
 '89aace00',
 '8b757ab8',
 'd45ed6a1',
 'ac92046e',
 '3bf1cf26'],
['3babcb9b',
 '7f0836bf',
 'ab3136ba',
 '15f99afc',
 '0d1da71f',
 'f7e47413',
 '69fdac0a',
 '2a512369',
 '9e4c8c7b',
 '15eb4a7d',
 'a1bbe385'],
['33505eae',
 '363d3849',
 'a29c5338',
 '6cf7d25c',
 '7ec0c298',
 '832735e1',
 '5154fc30',
 '0413e89d',
 'f71c4741',
 'f28c589a',
 '8d7e386c'],
['119b5b02',
 '5dc079d8',
 '26a5a3dd',
 '6f445b57',
 '29a42aea',
 '01ca3a3c',
 'e4d32835',
 'ab4ec3a4',
 'dcb1663e',
 '0ce40006'],
['4b5efe37',
 'abc5811c',
 '8d84fa81',
 '65abac75',
 '0086365d',
 '87d743c1',
 'fd20ea40',
 'f93fc684',
 'cc5087a3',
 '5be391b5',
 '7040c096'],
    ['b7dc8128', '15ba1109'],
    ['99abe2bb', 'd51b1749', '5a848010'],
    ['86ba578b', '4d6737eb'],
    ['e080a381', '92687c59', '19967db1'],
    ['c189aaf2', '4a09ace1', '99ea62f3'],
    ['53c6e11a', '9ed8f6da'],
    ['6088b756', 'd88ca108'],
    
['bcceccc6',
 'a7640a16',
 'f3cd5473',
 '7372e1a5',
 '76babcde',
 '6c517a88',
 '7dfe6d8a',
 'cf82af56',
 '587b5989',
 'a0faea5d',
 '5e3ea25a'],
 ['93edfe2e',
 '6043a2b4',
 '05ad839b',
 '6aeafed4',
 '6f8106d9',
 '6f4bd64e',
 '67aa2ada',
 'd3640339'],
 ['b1d5101d', '3b2048ee', 'ecc36b7f', '46b50ba8', '9e6b7fb5'],
 ['56817e2b',
 '16dffff1',
 '08fd73f3',
 'f5b8c21a',
 'b012cd7f',
 '00c73085',
 '3323d7e9',
 'ca11f653',
 '895865f3',
 '36fa3ebe',
 '1c178d24'],
 ['2c4e6db0',
 'f32856e4',
 '7cf1bc53',
 '63f13dd7',
 '26fd2d99',
 '7961e599',
 'f806dc10',
 '1beb320a',
 'c2baf0bd',
 'a592d54e',
 '1575e76c'],
 ['17ca3959', '4074bac2'],
['bc8f2793', 'd02b7a8e', 'd185d3ea', 'a2df0760'],
['d2659ab4'],
 ['804ee27f'],
 ['a1192f43'],
 ['3bb91dda'],
 ['85de926c'],
 ['a5e9da97'],
 ['2dc29e21'],
 ['262136f4'],
 ['8f094001'],
 ['c0415e5c'],
 ['b120f2ac'],
 ['1cc7cfca'],
 ['3d8c61b0'],
 ['8af75982'],
 ['792530f8'],
 ['14de4c5d'],
 ['28f975ea'],
 ['857f21c0'],
 ['74e5f8a7'],
 ['6bf9e3e1'],
 ['37db1c2f'],
 ['4ef8cdd3'],
 ['ad148f58'],
 ['5c3d2b2f'],
 ['86c924c4'],
 ['2a444e03'],
 ['e7561dd2'],
 ['37c53127'],
 ['dcaede90'],
 ['1996c610'],
 ['cfbd47c8'],
 ['86ba578b'],
 ['c1cac9a2', '3bb91ced'],
 ['dcb55a27'],
 ['1340b8d7'],
 ['c54cf6c5'],
 ['d06f75b5'],   
 ['c6971acf', '29f54413', '7d5c30a2', '06372577'],
 ['27253bdc'],
 ['27253bdc'],
 ['77261ab5'],
 ['b2dba42b'],
 ['1bb5fbdb'],
 ['1325467d'],
 ['37937459'],
 ['5e812b27'],
 ['c58186bf'],
 ['9ee1c98c'],
 ['84538528'],
 ['e64e2cfd'],
 ['49ed92e9'],
 ['bd701df8'],
 ['f50fc6c1'],
 ['d2e9262e'],
 ['2fb91ec1'],
 ['c952eb01'],
 ['a6d66e51'],
 ['71e712d8'],
 ['e7e44842'],
 ['4901243f'],
 ['beb0a7b9'],
 ['02a42007'],
 ['e694a35b'],
 ['b88f38da'],
 ['884228c8'],
 ['9b01374f'],
 ['56cd3b43'],
 ['a44b10dc'],
 ['bbfe0445'],
 ['5d042115'],
 ['de26c3a6'],
 ['598f4598'],
 ['fcfdffb6'],
 ['c7f7f0e1'],
 ['0a08139c'],
 ['e79f3763'],
 ['71fe8f75'],
 ['363c86c9'],
 ['022b4259'],
 ['565a3990'],
 ['d2278a3b'],
 ['b7530680'],
 ['67439901'],
 ['bb3e370b'],
 ['df4940d3'],
 ['90efca10'],
 ['d3f1e122'],
 ['e9c52111'],
 ['15a43e5b'],
 ['756e5507'],
 ['ea321fb1'],
 ['84b0e0c8'],
 ['4bb2f698'],
 ['56bcd38d'],
 ['499edb7c'],
 ['cdd22e43'],
 ['46cd75b4'],
 ['85d1b0de'],
 ['8d748b58'],
 ['47efca07'],
 ['5f5b2617'],
 ['9b23e8ee'],
 ['7ab78247'],
 ['736f9581'],
 ['b80e5e84'],
 ['4c2ec19f'],
 ['f54238ee'],
 ['47f43a44'],
 ['461eace6'],
 ['9e34ea74'],
 ['08ff79ad'],
 ['611485c5'],
 ['30df3273'],
 ['3a4be871'],
 ['2ec694de'],
 ['16667cc5'],
 ['7fd1ac25'],
 ['a8cc6fec'],
 ['003cd2ee'],
 ['1b54d27f'],
]

event_replacements = {}

all_events = set([])
for i, ev in enumerate(final_events):
    for e in ev:
        event_replacements[e] = f"event_{i}"
    all_events|=set(ev)

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


def make_round_counters(df, column):
    df = df[[c for c in df.columns if c in [column, "round", "game_session"]]].reindex(columns=[column, "round", "game_session"])
    df = df.groupby([column, "round", "game_session"])[column].count().to_frame(name="cnt").reset_index()
    return {k: v for k,v in df.groupby(column)["cnt"].mean().to_dict().items()}


def interactions(features):
    fnames = sorted(features.keys())
    result = {}
    for i, fa in enumerate(fnames):
        for j in range(i+1, len(fnames)):
            fb = fnames[j]
            fname = f"{fa}_interaction_{fb}"
            result[fname] = features[fa] / features[fb]
    return result

    
def process_log(df):
    current_session=df.game_session.iloc[-1]
    assessment_title=df.title.iloc[-1]   
    world=df.world.iloc[-1]

    history = df.iloc[:-1]
    movies = history[history.type=="Clip"]
    #history = history[history.type.isin(["Game", "Assessment"])]
    
    history = history[history.event_id.isin(all_events)].copy()
    
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
        return {k: v for k, v in result}, df
    
    
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
        stats=history[(history.title==game)&(history.event_id.isin(game_events[game]))]
        stats_features, stats = make_move_stats(stats, game)
        stats_features[f"{game}_event_code_count"] = stats.event_code.nunique()
        stats_features[f"{game}_event_id_count"] = stats.event_id.nunique()
        stats_features[f"{game}_session_id_count"] = stats.game_session.nunique()
        stats_features[f"{game}_n_actions"] = len(stats)
        result.update(stats_features)
        counters = {f"{game}_{k}": v for k, v in make_counters(stats, "event_id").items()}
        result.update(counters)
        #result.update(interactions(counters))
        result.update({f"{game}_{k}": v for k, v in make_counters(stats, "event_code").items()})
        return result
    
    for f in Parallel(n_jobs=cpu_count())(delayed(make_game_features)(game) for game in games):
        result.update(f)
    #for game in games:
    #    result.update(make_game_features(game))
    world_games = history[history.world==world]
    
    def make_world_features(game):
        result = {}
        stats=world_games[(world_games.title==game) & (world_games.event_id.isin(game_events[game]))]
        stats_features, stats=make_move_stats(stats, game)
        stats_features = {f"world_{k}": v for k, v in stats_features.items()}
        stats_features[f"world_{game}_event_code_count"] = stats.event_code.nunique()
        stats_features[f"world_{game}_event_id_count"] = stats.event_id.nunique()
        stats_features[f"world_{game}_session_id_count"] = stats.game_session.nunique()
        stats_features[f"world_{game}_n_actions"] = len(stats)
        result.update(stats_features)
        counters = {f"world_{game}_{k}": v for k, v in make_counters(stats, "event_id").items()}
        result.update(counters)
        #result.update(interactions(counters))
        result.update({f"world_{game}_{k}": v for k, v in make_counters(stats, "event_code").items()})
        return result
    world_games["event_id"] = world_games["event_id"].map(event_replacements)
    world_counters = {f"world_{k}": v for k, v in make_counters(world_games, "event_id").items()}
    result.update({f"world_{k}": v for k, v in make_counters(world_games, "event_code").items()})
    for f in Parallel(n_jobs=cpu_count())(delayed(make_world_features)(game) for game in games):
        result.update(f)
    #for game in games:
    #    result.update(make_world_features(game))
    #make_history_counters = partial(make_counters, history)
    history["event_id"] = history["event_id"].map(event_replacements)
    counters = make_counters(history, "event_id")
    result.update(counters)
    #result.update(interactions(counters))
    result.update(make_counters(history, "event_code"))
    #result.update(make_counters(movies, "title"))
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
        player_log = player_log.iloc[:(log_length + 1)]
        player_log["accuracy_group"] = accuracy_group
        player_log["target_game_session"] = game_session
        features = process_log(player_log)
        features["installation_id"] = installation_id
        features["accuracy_group"] = accuracy_group
        result.append(features)
    return pd.DataFrame(data=result).fillna(-1)


# In[ ]:


train_features = process_installations(train_labels, train, process_log)


# In[ ]:


#np.corrcoef(train_features["airshow_precision"], train_features["world_airshow_precision"])[0][1]


# In[ ]:


#train_features[[c for c in train_features.columns if isinstance(c, str) and ("crystalsrule" in c)]]


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
    
duplicate_features = get_duplicate_features(train_features, bad_features)

useful_features = [f for f in train_features.columns if f not in duplicate_features]
useful_features


# In[ ]:


train_features=train_features[useful_features].copy()


# In[ ]:


def lgb_quad_kappa(preds, true):
    true = true.get_label()
    #preds = preds.reshape((4, -1)).argmax(axis=0)
    preds = np.rint(preds)
    preds = np.maximum(0, preds)
    preds = np.minimum(3, preds)
    return "quad_kappa", quad_kappa(true, preds), True
    
    
def train_baseline(x_train,y_train, params=None):
    x_train_all, x_val_all,y_train_all,y_val_all = train_test_split(
        x_train,y_train,
        test_size=0.15,
        random_state=2019,
    )
    train_set = lgb.Dataset(x_train_all, y_train_all)
    val_set = lgb.Dataset(x_val_all, y_val_all)

#     params = {
#         'learning_rate': 0.01,
#         'bagging_fraction': 0.9,
#         'feature_fraction': 0.9,
#         'num_leaves': 14,
#         'lambda_l1': 0.1,
#         'lambda_l2': 1,
#         'metric': 'multiclass',
#         'objective': 'multiclass',
#         'num_classes': 4,
#         'random_state': 2019
#     }

    return lgb.train(params, train_set, num_boost_round=10000, early_stopping_rounds=2000, valid_sets=[val_set], verbose_eval=100)#,
                    #feval=lgb_quad_kappa)

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
test_features=process_test_installations(test)

for useful_feature in useful_features:
    if useful_feature not in test_features.columns:
        test_features[useful_feature]=-1
        print("Missing feature", useful_feature)

test_features=test_features[[c for c in useful_features if c != "accuracy_group"]].copy()


# In[ ]:


get_ipython().system('ls ../dsb2019/models')


# In[ ]:


with open("../dsb2019/models/regression_norm_params.json", "r") as f:
    validator_params=json.load(f)
selected_features = [f for f in train_features.columns if f not in ["installation_id", "game_session", "accuracy_group"]]
validator = adv_validation.AdversarialValidator(validator_params, train_features, test_features, selected_features=selected_features,test_size=0.5)
validator.fit()
print(validator.roc_auc())
validator.shap_important_features()


# In[ ]:


validator.lgb_important_features()


# In[ ]:


subtrain_installations=pd.Series(train_features.installation_id.unique()).sample(frac=1., random_state=2019)
subtrain_features=train_features[train_features.installation_id.isin(subtrain_installations.values)].copy()
def check_hyperparams(params):
    print(params)
    if "max_depth" in params:
        params["max_depth"] = int(params["max_depth"])
    if "num_leaves" in params:
        params["num_leaves"] = int(params["num_leaves"])

    train_baseline_with_params = partial(train_baseline, params=params)
    cv=InstallationFold(n_splits=3)
    predictions = cross_validate(subtrain_features, subtrain_features.accuracy_group, make_features_wrapper, train_baseline_with_params, make_predictions,
                                cv=cv)
    return {
        "loss": np.mean([mean_squared_error(true, pred) for pred, true in predictions]),
        "status": STATUS_OK,
        "params": params
    }


def tune(check_params, n_tries=25, n_learning_rate_tries=15, learning_rate=None, n_estimators=10_000):        
    if learning_rate is None:
        learning_rate_space = {
            'learning_rate': hp.loguniform("learning_rate", np.log(0.005), np.log(0.3)),
            'metric': 'rmse',
            'objective': 'rmse',
            #'num_classes': 4,
            'random_state': 2019,
            "n_estimators": n_estimators,

        }
        trials = Trials()
        result = fmin(check_params,
                      learning_rate_space, tpe.suggest, n_learning_rate_tries, trials=trials)
        print(result)
        learning_rate = round(trials.best_trial["result"]["params"]["learning_rate"], 3)

    param_space = {
        'metric': 'rmse',
        'objective': 'rmse',
        #'num_classes': 4,
        'lambda_l1': hp.uniform("lamba_l1", 1e-10, 1),
        'lambda_l2': hp.uniform("lambda_l2", 1e-10, 1),
        'random_state': 2019,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": hp.quniform("max_depth", 2, 16, 1),
        "num_leaves": hp.choice("num_leaves", [3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095]),
        "subsample": hp.quniform("subsample", 0.01, 1, 0.01),
        "feature_fraction": hp.quniform("feature_fraction", 0.01, 1, 0.01),
    }

    trials = Trials()
    fmin(check_params,
         param_space, tpe.suggest, n_tries, trials=trials)
    best_params = trials.best_trial["result"]["params"]
    return best_params


# In[ ]:


best_params=tune(check_hyperparams, n_tries=100, n_learning_rate_tries=10)


# In[ ]:


best_params


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


# best_params={'feature_fraction': 0.58,
#  'lambda_l1': 0.45619796864269707,
#  'lambda_l2': 0.033257384218246686,
#  'learning_rate': 0.007,
#  'max_depth': 14,
#  'metric': 'multiclass',
#  'n_estimators': 10000,
#  'num_classes': 4,
#  'num_leaves': 31,
#  'objective': 'multiclass',
#  'random_state': 2019,
#  'subsample': 0.9500000000000001}


# In[ ]:


with open("../dsb2019/models/regression_norm_params.json", "w") as f:
    json.dump(best_params, f)


# In[ ]:


baseline_model=train_baseline(train_features.drop(["installation_id", "accuracy_group"], axis=1), train_features.accuracy_group.values, 
               params=best_params)


# In[ ]:


predictions = cross_validate(train_features, train_features.accuracy_group, make_features_wrapper, partial(train_baseline, params=best_params), 
                             make_predictions)
print(np.mean([mean_squared_error(true, pred) for pred, true in predictions]), [mean_squared_error(true, pred) for pred, true in predictions])


# In[ ]:


baseline_model.save_model(str(MODELS_DIR / "regression_norm.lgb"))


# In[ ]:


features, target = make_features(train_features)
prediction=baseline_model.predict(features)
clf = ThresholdClassifier()
clf.fit(prediction, target)


# In[ ]:


print(clf.coef_)


# In[ ]:




