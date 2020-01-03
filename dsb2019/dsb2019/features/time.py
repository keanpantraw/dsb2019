import numpy as np


def make_calendar_features(df, assessment):
    ts = assessment.timestamp
    year = ts.year
    month = ts.month
    dayofweek = ts.dayofweek
    time = ts.time()
    return {
        "month": month,
        "dayofweek": dayofweek,
        "hour": time.hour,
    }


def make_base_time_features(df, assessment):
    start_end_times = df.groupby("game_session").agg({"timestamp": ["min", "max", "count"]}).reset_index()
    start_end_times.columns = ["game_session", "start_time", "end_time", "n_turns"]
    start_end_times["duration"] = start_end_times.end_time - start_end_times.start_time
    duration_minutes = start_end_times.duration / np.timedelta64(1, "m")
    result = {
        "mean_session_time_minutes": round(duration_minutes.mean(), 2), 
        "mean_session_turns":  round(start_end_times.n_turns.mean(), 2)
    }
    last_event_time = assessment.timestamp
    first_event_time = start_end_times.start_time.min()
    
    days_active = round((last_event_time - first_event_time) / np.timedelta64(1, "D"), 0) + 1
    result["games_per_day"] = round(df.game_session.nunique() / days_active, 2)
    result["games_played"] = df.game_session.nunique()
    minutes_between_games = ((start_end_times.start_time - start_end_times.start_time.shift(1)).dropna() / np.timedelta64(1, "m")).round(1)
    result["mean_minutes_between_games"] = round(minutes_between_games.mean(), 2)
    return result
