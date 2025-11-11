import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC   # linearSVC b/c it works better with large datasets, supports sparse output, and is for classification tasks
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

filteredDf = pd.read_csv('CSV_files/filtered_Df.csv')
randomizedDf = filteredDf.sample(frac = 1, random_state = 42)
randomizedDf = randomizedDf.reset_index(drop = True)
# must encode categorical columns to be numerical
categorical = ["developers", "publishers", "categories", "genres", "tags"]
numeric = ["price", "windows", "mac", "linux"]
# want to chop down number of individual/ unique values within the categorical columns, as after OHE it spat out 34000 columns (unworkable and crashed my laptop)
# one-hot categorical (sparse), pass numeric through
categorical_transformer = ColumnTransformer(transformers = [("ohe", OneHotEncoder(handle_unknown = 'ignore', sparse_output=True), categorical)],
    remainder='passthrough',   # numeric columns appended (left unchanged)
    sparse_threshold=0.0       # keep sparse output
)
# choose n_components for SVD (dimensionality reduction), TruncatedSVD works better for sparse output
svd = TruncatedSVD(n_components = 150, random_state=42) # 150 features max
# pipeline from scikit, makes fitting data EXTREMELY easy, especially when working with encoding for categorical columns. using over "model" method
pipeline = Pipeline([("ct", categorical_transformer), ("svd", svd), ("clf", LinearSVC())]) # essentially just allows me to put preprocessors (to cut down on the # of columns (one hot encoder creates 10s of thousands for my data)) in order to make the data more workable, max_iter = 1000 for logistic regression to balance convergence and perforance
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
print(f"Original shape: {randomizedDf.shape}")
X_ct = pipeline.named_steps['ct'].transform(X_train) # named_steps allows access to transformers
X_svd = pipeline.named_steps['svd'].transform(X_ct)
print(X_ct.shape, "after ColumnTransformer")    # printing shape after pipeline transformations
print(X_svd.shape, "after SVD")
# visualization
# general info about cleaned dataset
print(randomizedDf.info())
print(randomizedDf.describe())
print(randomizedDf['recommendation'].value_counts()) # counts how many of each output the column has (both 7000, I did this in the data cleaning step)
# confusion matrix tells which classes the model is confusing
disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred), display_labels = pipeline.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix | LinearSVC")
plt.show()
