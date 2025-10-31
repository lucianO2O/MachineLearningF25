import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

filteredDf = pd.read_csv('CSV_files/filtered_Df.csv')
randomizedDf = filteredDf.sample(frac = 1, random_state = 42)
randomizedDf = randomizedDf.reset_index(drop = True)
# must encode categorical columns to be numerical
categoricalColumns = ["developers", "publishers", "categories", "genres", "tags"]
numericColumns = ["price", "windows", "mac", "linux"]
encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)   # ignores unknowns
encoderTransform = pd.DataFrame(encoder.fit_transform(randomizedDf[categoricalColumns]), columns = encoder.get_feature_names_out(categoricalColumns))    # just transforms the columns into numerical data and then recovers the names, converts to dataframe
X = pd.concat([randomizedDf[numericColumns].reset_index(drop = True), encoderTransform.reset_index(drop = True)], axis = 1)     # concatenates numerical and categorical data transformed to numerical, resets the index and drops the old one
y = randomizedDf["recommendation"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12, shuffle=True)
model = GaussianNB()    # define model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model.score(X_test, y_test)
print(f"Accuracy = {model.score(X_test, y_test)}")
print(classification_report(y_test, y_pred))