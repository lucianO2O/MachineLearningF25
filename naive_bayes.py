import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

filteredDf = pd.read_csv('CSV_files/filtered_Df.csv')
randomizedDf = filteredDf.sample(frac = 1, random_state = 42)
randomizedDf = randomizedDf.reset_index(drop = True)
# must encode categorical columns to be numerical
categoricalColumns = ["developers", "publishers", "categories", "genres", "tags"]
numericColumns = ["price", "windows", "mac", "linux"]
encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)   # ignores unknown categories that it may not have seen during training when testing
encoderTransform = pd.DataFrame(encoder.fit_transform(randomizedDf[categoricalColumns]), columns = encoder.get_feature_names_out(categoricalColumns))    # just transforms the columns into numerical data and then recovers the names, converts to dataframe
# dropping some of the new columns as it is just too much to work with (~34000 columns created after encoding)
encoderTransform = encoderTransform.groupby("genres").filter(lambda x: len(x) > 25) # genre name must show up 25 times at least, dropping all other rows
endoderTransform = encoderTransform.groupby("categories").filter(lambda x: len(x) > 25) # same with categories
encoderTransform = encoderTransform.groupby("tags").filter(lambda x: len(x) > 25) # ...and with tags
X = pd.concat([randomizedDf[numericColumns].reset_index(drop = True), encoderTransform.reset_index(drop = True)], axis = 1)     # concatenates numerical and categorical data transformed to numerical, resets the index and drops the old one
y = randomizedDf["recommendation"]
# split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12, shuffle = True)
# define model
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model.score(X_test, y_test)
print(f"Accuracy = {model.score(X_test, y_test)}")
print(classification_report(y_test, y_pred))
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
disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred), display_labels = model.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix | Naive Bayes")
plt.show()

# get predicted probabilities
y_proba = model.predict_proba(X_test)

# plot the probability distribution for recommendation to see prediction trends
plt.hist(y_proba[:,1], bins=30, color='skyblue', edgecolor='black')
plt.title("Predicted Probability Distribution for Recommendation")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.show()
