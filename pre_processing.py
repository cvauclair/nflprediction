import pandas as pd
import numpy as np
from sklearn import preprocessing

def process_data(filename):
    raw = pd.read_csv(filename, index_col=0)

    # Get list of team abbreviations
    teams = np.union1d(raw['losing_abbr'].unique(), raw['winning_abbr'].unique())

    # Get features
    features = raw.columns[raw.columns.str.contains('home')].map(lambda s: s.replace('home_', ''))

    # Split in teams
    team_data = {}
    games_to_drop = []
    for team in teams:
        # Isolate team data
        team_raw_data = raw[(raw['losing_abbr'] == team) | (raw['winning_abbr'] == team)]
        
        # Get home/away data depending on game
        team_good_data = pd.DataFrame(np.where(pd.concat([pd.DataFrame(team_raw_data.index.str.contains(team.lower()))] * len(features), axis='columns'), 
                                                team_raw_data[team_raw_data.columns[team_raw_data.columns.str.contains('home')]], 
                                                team_raw_data[team_raw_data.columns[team_raw_data.columns.str.contains('away')]]), index=team_raw_data.index, columns=features)

        # Calculate running average of stats for each teams 
        team_data[team] = team_good_data.expanding().mean()

        # Make data the previous game's data and add the first game to the list of games to drop
        team_data[team] = team_data[team].shift(1)
        games_to_drop.append(team_data[team].index[0])

        team_data[team]['win'] = 0
        team_data[team]['tie'] = 0
        team_data[team]['loss'] = 0

    # Calculate difference between opposing teams and set result
    for index, row in raw.iterrows():
        winning_team = row['winning_abbr']
        losing_team = row['losing_abbr']

        # Calculate difference
        temp = team_data[losing_team].loc[index]
        team_data[losing_team].loc[index] -= team_data[winning_team].loc[index]
        team_data[winning_team].loc[index] -= temp

        # Check for tie 
        if row['home_points'] == row['away_points']:
            team_data[winning_team].loc[index, 'tie'] = 1
            team_data[losing_team].loc[index, 'tie'] = 1

        # Set result win/loss
        team_data[winning_team].loc[index,'win'] = 1
        team_data[losing_team].loc[index, 'loss'] = 1

    # Combine data for all games and drop first games
    all_data = pd.concat([team_data[team] for team in teams])
    all_data.drop(games_to_drop, inplace=True)

    return all_data

def standardize_data(data):
    # Standarize data (skip win/tie/loss)
    sd_scaler = preprocessing.StandardScaler()
    standardized_data = pd.DataFrame(sd_scaler.fit_transform(data.drop(['win', 'tie', 'loss'], axis='columns')),
                                     index=data.index, 
                                     columns=data.columns.drop(['win', 'tie', 'loss']))

    # Add back win/tie/loss
    standardized_data['win'] = data['win']
    standardized_data['tie'] = data['tie']
    standardized_data['loss'] = data['loss']

    # normalized_data.to_csv("data/1966_processed.csv")
    return standardized_data

processed = []
for season in range(1966, 1975):
    processed.append(process_data('data/raw/{}_games.csv'.format(season)))

all_processed = pd.concat(processed)
standardized_data = standardize_data(all_processed)
print(standardized_data)
standardized_data.to_csv('data/processed.csv')
# print(process_data('data/raw/1966_games.csv'))