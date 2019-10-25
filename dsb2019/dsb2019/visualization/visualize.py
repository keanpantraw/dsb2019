def show_game_sessions(df, installation_id, filter_cols=tuple()):
    df = df[df.installation_id == installation_id].copy()
    game_sessions = df.game_session.unique()
    for game_session in game_sessions:
        session = df[df.game_session==game_session].copy()
        display(Markdown(describe_session(game_session, session)))
        with pd.option_context('display.max_rows', 999, 'display.max_columns', 999, 
                               'display.max_colwidth', 999):
            display(format_session(session, filter_cols))


def describe_session(game_session, session):
    head = session.sort_values("event_count").head(1)
    if "version" in json.loads(head.event_data.iloc[0]):
        version = json.loads(head.event_data.iloc[0])["version"]
    else:
        version = "-"
    return f"__{game_session} {head.title.iloc[0]} {head.type.iloc[0]} {head.world.iloc[0]} v{version}__ "


def format_session(session, filter_cols):
    session = session.sort_values("event_count")
    session["timestamp"]=session["timestamp"].str[:19]
    session["event_data"] = session.event_data.apply(json.loads)
    columns = sorted(reduce(set.__or__, session.event_data.apply(lambda d: set(d.keys())).values))
    for c in columns:
        session[c] = session.event_data.apply(lambda x: x.get(c, '-'))
    session["version"]=None
    session = session.drop(["game_session", "installation_id", "event_data", "game_time", 
                            "title", "type", "world", "version"] + [c for c in filter_cols if c in session.columns], axis=1)
    session.set_index("event_count", inplace=True)
    
    return session