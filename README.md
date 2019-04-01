NFL Machine learning predictions

# Description
Using game level data scrapped from www.pro-football-reference.com (using the python package `sportsreference` https://pypi.org/project/sportsreference/), the goal was to predict the outcome (win, tie or loss) of a game between team A and team B with data that would be available (in real life) before the game. 

The raw data collected for each game consists of the following features (can be found under `data/raw/`):
```
['first_downs', 'fourth_down_attempts', 'fourth_down_conversions', 'fumbles', 'fumbles_lost', 'interceptions', 'net_pass_yards', 'pass_attempts', 'pass_completions', 'pass_touchdowns', 'pass_yards', 'penalties', 'points', 'rush_attempts', 'rush_touchdowns', 'rush_yards', 'third_down_attempts', 'third_down_conversions', 'times_sacked', 'total_yards', 'turnovers', 'yards_from_penalties', 'yards_lost_from_sacks']
```

From there, a moving average was applied to the game data for each season of every team. For each game, the game data was summed to the game data of the other games of the same team and season and then divided by the number of games that particular team had completed this particular season. In other words, the processed data was the 'team average to date' of the raw game stats. 

The data was also shifted so that the result (win/loss/tie) was associated with the cummulative data up until the previous game.

Then for every match up between a team A and a team B, the average stats to date of team A were substracted by team B's stats so as to have the difference between both teams of their average stats. By convention, the result is shown with respect to team A (e.g.: a 'loss' result indicates that team A has lost and team B has won).

Finally, the differences in average stats to date were standardized.

The final processed data can be found at `data/processed.csv` (note that the first column is the code of the game, and the `team` column is the name of team A).

# Dependencies
This project depends on the following python packages: `pytorch`, `pandas`, `numpy`, `scikit-learn` and `sportsreference`.

# Running
## Scrapping
To get the raw data, run `python scrapper.py` (the seasons for which the data will be scrapped can be changed by modifying the `season_start` and `season_end` fields in the script)

## Processing
To process the raw data, run `python pre_processing.py` (as with the scrapping script, the seasons for which the raw data will be processed can be changed by modifying the `season_start` and `season_end` fields in the script).

## Model training
To train the model, run `python main.py`.