import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Display available input files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load datasets
train = pd.read_csv('../input/spaceship-titanic/train.csv')
test = pd.read_csv('../input/spaceship-titanic/test.csv')

# Data Preprocessing
train.drop(["Name", "PassengerId"], axis='columns', inplace=True)
test.drop(["Name", "PassengerId"], axis='columns', inplace=True)

# Encode categorical data
le = LabelEncoder()
train['HomePlanet'] = le.fit_transform(train['HomePlanet'])
train['CryoSleep'] = le.fit_transform(train['CryoSleep'])
train['Cabin'] = le.fit_transform(train['Cabin'])
train['Transported'] = le.fit_transform(train['Transported'])

# Compute total expenses and drop individual expense columns
train['expenses'] = train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis='columns')
train.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis='columns', inplace=True)
train.drop(['VIP', 'Destination'], axis='columns', inplace=True)

# Define features and fill missing values
features = ['Cabin', 'Age', 'expenses', 'CryoSleep']
train.fillna(0, inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(train[features], train['Transported'], train_size=0.4)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=50)
model.fit(X_train, y_train)

# Evaluate model
print("Model Accuracy:", model.score(X_test, y_test))

# Process test data
test['HomePlanet'] = le.fit_transform(test['HomePlanet'])
test['CryoSleep'] = le.fit_transform(test['CryoSleep'])
test['Cabin'] = le.fit_transform(test['Cabin'])
test['expenses'] = test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis='columns')
test.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'VIP', 'Destination'], axis='columns', inplace=True)
test.fillna(0, inplace=True)

# Generate predictions
predictions = model.predict(test)

# Save predictions to a CSV file
output = pd.DataFrame({
    'PassengerId': pd.read_csv('../input/spaceship-titanic/test.csv').PassengerId,
    'Transported': predictions
})
output['Transported'] = output['Transported'].replace({0: False, 1: True})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
