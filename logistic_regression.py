import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc

filteredDf = pd.read_csv('CSV_files/filtered_Df.csv')
randomizedDf = filteredDf.sample(frac = 1, random_state = 42)
randomizedDf = randomizedDf.reset_index(drop = True)
# must encode categorical columns to be binary
lists = ["categories", "genres", "tags"] # columns that have lists as values
strings = ["developers", "publishers"] # columns with strings as values
numerics = ["price", "windows", "mac", "linux"]

mlbDfs = []
for column in lists:
    mlb = MultiLabelBinarizer(sparse_output = False)
    randomizedDf[column] = randomizedDf[column].str.replace("[", "") # removing all brackets and single quotes to not interfere with retrieving names for new columns
    randomizedDf[column] = randomizedDf[column].str.replace("]", "")
    randomizedDf[column] = randomizedDf[column].str.replace("'", "")
    randomizedDf[column] = randomizedDf[column].str.replace("-", "")
    matrix = mlb.fit_transform(randomizedDf[column].str.split(', '))           # matrix created, split on comma for names
    dfColumn = pd.DataFrame(matrix, columns = [f"{column}__{label}" for label in mlb.classes_]) # meaningful column names for each column generated
    mlbDfs.append(dfColumn)     # appends each column in dataframe to one dataframe
listsTransformed = pd.concat(mlbDfs, axis = 1) # concatenates the column dataframes to one dataframe

# genres, categories, and tags have duplicate values (tags__action, genres__action, etc), want to drop the non-genre duplicates
priorityOrder = ['genres', 'tags', 'categories'] # want to order by genres, so tags and categories drop the dupes
sortedColumns = sorted(listsTransformed.columns, key = lambda x: priorityOrder.index(x.split("__")[0])) # sorted dataframe by priority order based on column prefix ("genres", "tags", "categories")
listsTransformed = listsTransformed[sortedColumns] # reorder by priority

seenLabels = [] # empty list to hold labels that have been seen (in genres)
columnsToKeep = []
for column in listsTransformed.columns:
    label = column.split("__")[1]  # the suffix ("action", "adventure")
    if label not in seenLabels:
        columnsToKeep.append(column)
        seenLabels.append(label)
listsTransformed = listsTransformed[columnsToKeep]

# OHE
ohe = OneHotEncoder(handle_unknown = "ignore", sparse_output = False)
stringsTransformed = pd.DataFrame(ohe.fit_transform(randomizedDf[strings]), columns = ohe.get_feature_names_out(strings)) # converts string columns to binary columns, converts that to a dataframe to be concatenated with lists and numerics, gets meaningful names for features

# choose n_components for SVD (dimensionality reduction), TruncatedSVD works better for large data
# pipeline from scikit, makes fitting data very easy with SVD, using over "model"
pipeline = Pipeline([("svd", TruncatedSVD(n_components = 150, random_state=42)),    # 150 features
                     ("clf", LogisticRegression(max_iter = 2000, C = 10.0, class_weight = 'balanced'))]) # max_iter = 2000 for logistic regression to balance convergence and performance
X = pd.concat([stringsTransformed, listsTransformed, randomizedDf[numerics]], axis = 1) # axis = 1 means columns
y = randomizedDf['recommendation']
# split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12, shuffle = True)
pipeline.fit(X_train, y_train)  # fit pipeline to training data
y_pred = pipeline.predict(X_test)
y_pred_train = pipeline.predict(X_train)

pipeline.score(X_test, y_test)
pipeline.predict(X_test)

print(f"Accuracy = {pipeline.score(X_test, y_test)}")
print("Classification Report for training:\n", classification_report(y_train, y_pred_train))
print("Classification Report for testing:\n", classification_report(y_test, y_pred))

print(randomizedDf['recommendation'].value_counts()) # counts how many of each output the column has
X_svd = pd.DataFrame(pipeline.named_steps['svd'].transform(X_train))
print(f"Original shape: {randomizedDf.shape}")
print(f"Shape after SVD: {X_svd.shape}")
# visualization
# confusion matrix tells which classes the model is confusing
disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred), display_labels = pipeline.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix | Logistic Regression")
plt.show()

# calculate ROC curve
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
# plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Steam Recommendation Classification')
plt.legend()
plt.show()

# plot the probability distribution for recommendation to see prediction trends, essentially a confidence scale from 0 being least confident that the output = 1 (recommended) to 1 being most confident
# separate probabilities by actual class
proba_recommended = y_pred_proba[y_test == 1]
proba_not_recommended = y_pred_proba[y_test == 0]

# plot overlapping histograms to show how confident model was
plt.figure(figsize=(8,5))
plt.hist(proba_recommended, bins=30, alpha=0.6, color='green', label='Actual Recommended (1)', edgecolor='black')
plt.hist(proba_not_recommended, bins=30, alpha=0.6, color='red', label='Actual Not Recommended (0)', edgecolor='black')

plt.title('Predicted Probability Distribution by Target Output')
plt.xlabel('Predicted Probability of Recommendation')
plt.ylabel('Frequency')
plt.legend()
plt.show()

categorical = pd.concat([listsTransformed, stringsTransformed], axis = 1)
top15 = categorical.sum().sort_values(ascending=False).head(15)
top15_features = top15.index
corr_top15 = X[top15_features].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_top15, annot=False, cmap='viridis')
plt.title("Correlation Heatmap (Top 10 Features)")
plt.tight_layout() # fits graph with parameters to screen
plt.show()