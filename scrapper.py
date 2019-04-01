from sportsreference.nfl.boxscore import Boxscore
from sportsreference.nfl.boxscore import Boxscores
from sportsreference.nfl.teams import Teams
import pandas as pd

def get_games_data(season):
    games_data = []
    weeks = Boxscores(start_week, season, end_week)
    for key in weeks.games:
        for game in weeks.games[key]:
            print("Fetching {} data".format(game['boxscore']))
            games_data.append(Boxscore(game['boxscore']).dataframe)

    return pd.concat(games_data)
    
# def get_teams_abbr(season):
#     print("Fetching {} teams".format(season))
#     teams = Teams(season)
#     return teams.dataframes.index.to_series()

start_week = 1
end_week = 22
season_start = 1976
season_end = 1996

for season in range(season_start, season_end):
    print("---------- Season {} ----------".format(season))
    # teams = get_teams_abbr(season)
    # teams.to_csv("data/raw/{}_teams.csv".format(season))
    data = get_games_data(season)
    data.to_csv("data/raw/{}_games.csv".format(season))