import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

# Calculate accuracy for training set
accuracy_train = accuracy_score(y_train, y_train_pred)
print("\nAccuracy (Training Set):", accuracy_train)

# Calculate accuracy for test set
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Accuracy (Test Set):", accuracy_test)
