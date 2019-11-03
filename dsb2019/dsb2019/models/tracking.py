import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from pytz import timezone


TRACKING_FILE = Path(os.getenv("TRACKING_FILE", "/code/dsb2019/tracking.csv"))
TZ = timezone("Europe/Samara")


def track_experiment(name: str, mean_score: float, cv_scores: np.array, notebook_path: str) -> pd.DataFrame:
    experiment = pd.DataFrame(data={"name": [name], "time": [datetime.now(TZ)], "mean_score": [mean_score], "cv_scores": [cv_scores], 
                                    "notebook_path": [notebook_path], "submission_path": [None], "submission_score": [None]})
    if TRACKING_FILE.exists():
        tracking_df = pd.read_csv(TRACKING_FILE)
        if list(tracking_df.columns) != list(experiment.columns):
            raise RuntimeError("Tracking schema changed!")
        experiment_mask = tracking_df.name==name 
        if experiment_mask.sum() > 0:
            experiment.index = tracking_df.index[experiment_mask]
            tracking_df.update(experiment)
        else:
            tracking_df = tracking_df.append(experiment, ignore_index=True)
    else:
        tracking_df = experiment
    tracking_df.to_csv(TRACKING_FILE, index=False)
    return tracking_df


def track_submission_info(name: str, submission_path: str, submission_score: float) -> pd.DataFrame:
    if not TRACKING_FILE.exists():
        raise RuntimeError(f"{TRACKING_FILE} doesn't exist")
    tracking_df = pd.read_csv(TRACKING_FILE)
    experiment_mask = tracking_df.name.isin([name])
    if experiment_mask.sum() == 0:
        raise RuntimeError(f"Experiment {name} wasn't tracked")
    tracking_df.loc[experiment_mask, "submission_path"] = submission_path
    tracking_df.loc[experiment_mask, "submission_score"] = submission_score
    tracking_df.to_csv(TRACKING_FILE, index=False)
    return tracking_df
