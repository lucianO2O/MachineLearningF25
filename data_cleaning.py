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
# want to change empty cells to null, they are currently strings with nothing inside
for column in gamesData.columns:    # loops through all columns
    cleanedColumn = []      # temp set for cleaned values
    for cell in gamesData[column]:      # loops through each specific cell in column
        if (pd.isna(cell) or
                isinstance(cell, str) and cell.strip() in ("[]", "{}", "") or
                isinstance(cell, list) and len(cell) == 0):  # if the value is already NaN OR take the cell, remove any spaces around it, and then check to see if string matches "[]", "{}", ""
            cleanedColumn.append(np.nan)    # convert the empty cell to an ACTUAL null
        else: cleanedColumn.append(cell)    # else just append it
    gamesData[column] = cleanedColumn   # set column equal to the new cleaned one

# want to convert categorical columns from string looking lists to actual lists
categoricalColumns = ["categories", "genres", "tags", "developers", "publishers"]    # empty set to store the columns i want to clean
cleanedColumn = []
for column in gamesData.categoricalColumns:
    for cell in gamesData[column]:
        if cell in ('None', 'NaN', 'nan', ''):
            cleanedColumn.append(np.nan)      # if NaN already, then just append to this new empty set
        elif isinstance(cell, str) and cell.startswith('{') and cell.endswith('}') or cell.startswith('[') and cell.endswith(']'):     # make sure it's a string
            try:
                cleanedColumn.append(list(ast.literal_eval(cell)))   # literal eval turns the cell from a literal into a python object, and the list()
                                                                   # function turns it into, a more usable, list, which then
                                                                   # gets appended to the new column
            except (ValueError, SyntaxError):   # good practice to have exception with ast.literal_eval for safety issues
                cleanedColumn.append(np.nan)
        else: cleanedColumn.append(cell)
    gamesData[column] = cleanedColumn
# want to remove # of people who voted for tags in the column and just leave the tag
gamesData['tags'] = gamesData['tags'].str.replace(r'[0-9:]', '', regex=True) # gets rid of the numbers and colon
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
# randomly selects entries with over 100 reviews, splitting them equally in half based on target variable
num_pos = revFilter[revFilter['recommendation'] == 1].shape[0] # number of rows w/ output = 1
num_neg = revFilter[revFilter['recommendation'] == 0].shape[0] # number of rows w/ output = 0
n = min(num_pos, num_neg) # takes minimum of both
filteredDf = pd.concat([
    revFilter[revFilter['recommendation'] == 1].sample(n, random_state=42), # ensures both classes have same number of rows
    revFilter[revFilter['recommendation'] == 0].sample(n, random_state=42)
])
filteredDf.to_csv('CSV_files/filtered_Df.csv', index = False)

print(filteredDf.shape)