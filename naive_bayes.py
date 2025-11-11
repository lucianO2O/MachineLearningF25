import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

filteredDf = pd.read_csv('CSV_files/filtered_Df.csv')
randomizedDf = filteredDf.sample(frac = 1, random_state = 42)
randomizedDf = randomizedDf.reset_index(drop = True)
# must encode categorical columns to be numerical
categorical = ["developers", "publishers", "categories", "genres", "tags"]
numeric = ["price", "windows", "mac", "linux"]
# want to chop down number of individual/ unique values within the categorical columns, as after OHE it spat out 34000 columns (unworkable and crashed my laptop)
# one-hot categorical (sparse), pass numeric through
categorical_transformer = ColumnTransformer(transformers=[("ohe", OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical)],
    remainder='passthrough',   # numeric columns appended
    sparse_threshold=0.0       # keep sparse output
)
# choose n_components for SVD (dimensionality reduction), TruncatedSVD works better for sparse output
svd = TruncatedSVD(n_components = 150, random_state=42) # 150 features max
# pipeline from scikit, makes fitting data EXTREMELY easy, especially when working with encoding for categorical columns. using over "model"
pipeline = Pipeline([("ct", categorical_transformer), ("svd", svd), ("clf", GaussianNB())]) # essentially just allows me to put preprocessors to cut down on the # of columns (one hot encoder creates 10s of thousands for my data) in order to make the data more workable
X = randomizedDf[categorical + numeric]
y = randomizedDf['recommendation']
# split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12, shuffle = True)
pipeline.fit(X_train, y_train)  # fit pipeline to training data
y_pred = pipeline.predict(X_test)
pipeline.score(X_test, y_test)
pipeline.predict(X_test)
print(f"Accuracy = {pipeline.score(X_test, y_test)}")
print(classification_report(y_test, y_pred))
print("Original shape:", randomizedDf.shape)
X_ct = pipeline.named_steps['ct'].transform(X_train)
X_svd = pipeline.named_steps['svd'].transform(X_ct)
print(X_ct.shape, "after ColumnTransformer")    # printing shape after pipeline transformations
print(X_svd.shape, "after SVD")
# visualization
# general info about cleaned dataset
print(randomizedDf.info())
print(randomizedDf.describe())
print(randomizedDf['recommendation'].value_counts()) # counts how many of each output the column has (both 7000, I did this in the data cleaning step)

# plot recommendations
randomizedDf['recommendation'].value_counts().plot(kind = 'bar', color = 'skyblue') # plots the # of recommendations per 1/0
plt.title('Distribution of recommendations') #  labels and titles
plt.xlabel('Recommendation')
plt.ylabel('Count')
plt.show()

# want to see how price relates to recommendation
sns.boxplot(x = 'recommendation', y = 'price', data = randomizedDf)
plt.title('Recommendation vs. Price')
plt.xlabel('Recommendation')
plt.ylabel('Price')
plt.show()

# confusion matrix tells which classes the model is confusing
disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred), display_labels = pipeline.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix | Naive Bayes")
plt.show()

# plot the probability distribution for recommendation to see prediction trends, essentially a confidence scale from 0 being least confident that the output = 1 (recommended) to 1 being most confident
y_proba = pipeline.predict_proba(X_test)[:, 1]  # if the game is recommended (output = 1)

# separate probabilities by actual class
proba_recommended = y_proba[y_test == 1]
proba_not_recommended = y_proba[y_test == 0]

# plot overlapping histograms to show how confident model waas
plt.figure(figsize=(8,5)) # don't want anything being cut off, big figure
plt.hist(proba_recommended, bins=30, alpha=0.6, color='green', label='Actual Recommended (1)', edgecolor='black')
plt.hist(proba_not_recommended, bins=30, alpha=0.6, color='red', label='Actual Not Recommended (0)', edgecolor='black')

plt.title('Predicted Probability Distribution by Target Output')
plt.xlabel('Predicted Probability of Recommendation')
plt.ylabel('Frequency')
plt.legend()
plt.show()