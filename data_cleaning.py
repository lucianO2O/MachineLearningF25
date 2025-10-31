import pandas as pd
import numpy as np
import ast

# read csv file
gamesData = pd.read_csv('CSV_files/games_march2025_full.csv')
# show all columns when printing
pd.set_option('display.max_columns', None)

# get rid of any duplicate rows
gamesData = gamesData.drop_duplicates()
# get rid of playtest games that have no data
gamesData = gamesData[~gamesData['name'].str.contains('playtest', case = False, na = False)]
# want to change empty cells to null, they are currently strings with nothing inside in the dataset
for column in gamesData.columns:    # loops through all columns
    cleanedColumn = []      # temp set for cleaned values
    for cell in gamesData[column]:      # loops through each specific cell in column
        if pd.isna(cell) or isinstance(cell, str) and cell.strip() in ("[]", "{}", "") or isinstance(cell, list) and len(cell) == 0:  # if the value is already NaN OR take the cell, remove any spaces around it, and then check to see if string matches "[]", "{}", ""
            cleanedColumn.append(np.nan)    # convert the empty cell to an ACTUAL null
        else: cleanedColumn.append(cell)    # else just append it
    gamesData[column] = cleanedColumn   # set column equal to the new cleaned one
# want to remove # of people who voted for tags in the column and just leave the tag
gamesData['tags'] = gamesData['tags'].str.replace(r'[0-9:]', '', regex=True) # gets rid of the numbers and colon
# want to convert tags column from string looking lists to actual lists
cleanedTags = []    # empty set to store the new cleaned column
for cell in gamesData['tags']:
    if cell in ('None', 'NaN', 'nan' ''):
        cleanedTags.append(np.nan)      # if NaN already, then just append to this new empty set
    elif isinstance(cell, str) and cell.startswith('{') and cell.endswith('}'):     # make sure it's a string
        try:
            cleanedTags.append(list(ast.literal_eval(cell)))   # literal eval turns the cell from a string that
                                                               # looks like a literal, into a python object, and the list()
                                                               # function turns it into, a more usable, list, which then
                                                               # gets appended to the new column
        except (ValueError, SyntaxError):   # good practice to have exception with ast.literal_eval for safety issues
            cleanedTags.append(np.nan)
    else: cleanedTags.append(cell)
gamesData['tags'] = cleanedTags     # assign cleaned values to the column
# get rid of any unnecessary columns
gamesData = gamesData.drop(columns = {'appid', 'release_date', 'dlc_count', 'required_age',
                                      'detailed_description', 'about_the_game',
                                      'short_description', 'reviews',
                                      'header_image', 'website', 'support_url',
                                      'support_email', 'metacritic_url', 'achievements',
                                      'recommendations', 'notes', 'supported_languages',
                                      'full_audio_languages', 'packages',
                                      'user_score', 'score_rank',
                                      'estimated_owners', 'average_playtime_forever',
                                      'average_playtime_2weeks', 'median_playtime_forever',
                                      'median_playtime_2weeks', 'peak_ccu', 'pct_pos_recent',
                                      'screenshots', 'movies', 'num_reviews_recent', 'discount',
                                      'metacritic_score', 'positive', 'negative' })     # I don't want to train my algorithm with any of these columns (missing lots of data, unusable, doesn't exactly correlate to a recommendation, no image processing/ links)
# drop all rows that have empty values in any column
gamesData = gamesData.dropna()
# rename column to reflect entries, being the percentage of positive reviews out of total reviews
gamesData = gamesData.rename(columns = {'pct_pos_total': 'pos_rev_pct'})
# create target column recommendation to determine if a game is liked by the community or not
gamesData['recommendation'] = (gamesData['pos_rev_pct'] >= 80).astype(int) # percentage of positive reviews is greater or equal to 80 (the threshold for games with steam ratings 'Very Positive' or above. must have at least 50 positive reviews)
# make a filter to chop down number of entries
revFilter = gamesData[gamesData['num_reviews_total'] >= 100]
# randomly selects entries with over 100 reviews equaling to 14k, splitting them equally in half based on target variable
filteredDf = pd.concat([
    revFilter[revFilter['recommendation'] == 1].sample(7000, random_state=42), # essentially creating two dataframes and then combining them (.concat)
    revFilter[revFilter['recommendation'] == 0].sample(7000, random_state=42) ])
filteredDf.to_csv('filtered_Df.csv', index = False)

print(filteredDf.shape)