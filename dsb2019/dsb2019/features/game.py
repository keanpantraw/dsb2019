games = ['Scrub-A-Dub', 'All Star Sorting', 'Mushroom Sorter (Assessment)',
       'Air Show', 'Crystals Rule', 'Bird Measurer (Assessment)',
       'Dino Drink', 'Bubble Bath', 'Dino Dive', 'Chow Time',
       'Cauldron Filler (Assessment)', 'Pan Balance', 'Happy Camel',
       'Cart Balancer (Assessment)', 'Chest Sorter (Assessment)',
       'Leaf Leader']


def calculate_ratios(df):
    n_correct=df.correct_move.sum()
    n_incorrect=df.wrong_move.sum()
    ratio=n_correct/(n_correct+n_incorrect)
    return n_correct, n_incorrect, ratio


def assessment_title(df):
    assessment_title=df.title.iloc[-1]    
    return {"title": games.index(assessment_title)}


def make_move_stats(df, title="", n_lags=2):
    if "correct" in df.columns:
        df["correct_move"] = df.correct == True
        df["wrong_move"] = df.correct == False
    else:
        df["correct_move"]=False
        df["wrong_move"]=False
    result = []
    result.extend(zip([f"n_correct {title}", f"n_incorrect {title}", f"global_ratio {title}"], calculate_ratios(df)))
    if n_lags:
        last_sessions = df.game_session.unique()[-n_lags:]
        for i in range(n_lags):
            if i < len(last_sessions): 
                result.extend(zip([f"n_correct {title} {i}", f"n_incorrect {title} {i}",f"ratio {title} {i}"], 
                                    calculate_ratios(df[df.game_session==last_sessions[i]])))
            else:
                result.extend(zip([f"n_correct {title} {i}", f"n_incorrect {title} {i}",f"ratio {title} {i}"], [None, None, None]))
    return {k: v for k, v in result}
