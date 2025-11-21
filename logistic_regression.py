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
# must encode categorical columns to be numerical
lists = ["categories", "genres", "tags"] # columns that have lists as values
strings = ["developers", "publishers"] # columns with strings as values
numerics = ["price", "windows", "mac", "linux"]

mlbDfs = []
for column in lists:
    mlb = MultiLabelBinarizer(sparse_output = False)
    matrix = mlb.fit_transform(randomizedDf[column])           # sparse (n_samples, n_labels_for_column)
    dfColumn = pd.DataFrame(matrix, columns = [f"{column}__{label}" for label in mlb.classes_]) # meaningful column names for each column
    mlbDfs.append(dfColumn)     # appends each column in dataframe to one dataframe
listsTransformed = pd.concat(mlbDfs, axis = 1)

ohe = OneHotEncoder(handle_unknown = "ignore", sparse_output = False)
stringsTransformed = pd.DataFrame(ohe.fit_transform(randomizedDf[strings]), columns = ohe.get_feature_names_out(strings)) # specifically for sparse matrices, converts to DF

# choose n_components for SVD (dimensionality reduction), TruncatedSVD works better for sparse output
# pipeline from scikit, makes fitting data very easy with SVD, using over "model"
pipeline = Pipeline([("svd", TruncatedSVD(n_components = 150, random_state=42)),    # 150 features
                     ("clf", LogisticRegression(max_iter=1000))]) # max_iter = 1000 for logistic regression to balance convergence and performance
X = pd.concat([stringsTransformed, listsTransformed, randomizedDf[numerics]], axis = 1)
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
# general info about cleaned dataset
print(randomizedDf.info())
print(randomizedDf.describe())
print(randomizedDf['recommendation'].value_counts()) # counts how many of each output the column has (both 7000, I did this in the data cleaning step)
X_svd = pipeline.named_steps['svd'].transform(X_train)
print(f"Original shape: {randomizedDf.shape}")
print(X_svd.shape, "shape after SVD")
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