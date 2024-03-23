import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the csv file
df = pd.read_csv("allocated_values3.csv")

print(df.head())

# Select independent and dependent variable
X = df[["BMI", "Cycle(R/I)", "FSH/LH", "PRL(ng/mL)"]]
y = df["Allocated Values"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("MlModel.pkl", "wb"))

# Predictions on training set
y_train_pred = classifier.predict(X_train)

# Predictions on test set
y_test_pred = classifier.predict(X_test)

# Calculate mean squared error for training set
mse_train = mean_squared_error(y_train, y_train_pred)
print("\nMean Squared Error (Training Set):", mse_train)

# Calculate R^2 score for training set
r2_train = r2_score(y_train, y_train_pred)
print("R^2 Score (Training Set):", r2_train)

# Calculate mean squared error for test set
mse_test = mean_squared_error(y_test, y_test_pred)
print("\nMean Squared Error (Test Set):", mse_test)

# Calculate R^2 score for test set
r2_test = r2_score(y_test, y_test_pred)
print("R^2 Score (Test Set):", r2_test)
