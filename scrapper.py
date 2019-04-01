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
season_start = 1966
season_end = 1976

for season in range(season_start, season_end):
    # teams = get_teams_abbr(season)
    # teams.to_csv("data/raw/{}_teams.csv".format(season))
    data = get_games_data(season)
    data.to_csv("data/raw/{}_games.csv".format(season))


# team_data = {}
# print("Splitting raw data")
# for game in games:

# season = 2016
# teams = Teams(season)

# Init database
# team_data = {}
# for team in teams:
#     team_data[team.abbreviation] = pd.DataFrame()

# features_set = False
# features = []
# for team in teams:
#     print("Fetching schedule data for {}".format(team.abbreviation))
#     games_data = team.schedule.dataframe_extended
#     games_data.to_csv('data/raw_{}_{}.csv'.format(team.abbreviation, season))
    
#     # Set features if not set already
#     if not features_set:
#         features = data.columns[data.columns.str.contains('away')].map(lambda s: s.replace('away_',''))
#         features_set = True

#     # Split data into away and home
#     # home_games_data = games_data[games_data.index.str.contains(team.abbreviation)]
#     # away_games_data = games_data[~games_data.index.str.contains(team.abbreviation)]
    
#     # team_data[team.abbreviation].columns = data.columns[data.columns.str.contains('home')].map(lambda s: s.replace('home_', ''))

#     print("Getting features for {}".format(team.abbreviation))
#     team_data[team.abbreviation] = pd.DataFrame(np.where(pd.concat([pd.DataFrame(games_data.index.str.contains(team.abbreviation))] * len(features), axis='columns'), 
#                                                 games_data[games_data.columns[games_data.columns.str.contains('home')]], 
#                                                 games_data[games_data.columns[games_data.columns.str.contains('away')]]), columns=features)


# print(team_data['nwe'])