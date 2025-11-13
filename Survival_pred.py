import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Preprocess both datasets
# Fill missing values
train["Age"].fillna(train["Age"].median(), inplace=True)
test["Age"].fillna(test["Age"].median(), inplace=True)

train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)
test["Fare"].fillna(test["Fare"].median(), inplace=True)

# Convert text to numbers
train = pd.get_dummies(train, columns=["Sex", "Embarked"], drop_first=True)
test = pd.get_dummies(test, columns=["Sex", "Embarked"], drop_first=True)

# Align columns between train and test
X_train = train[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_male", "Embarked_Q", "Embarked_S"]]
y_train = train["Survived"]

X_test = test[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_male", "Embarked_Q", "Embarked_S"]]

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test.csv
predictions = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})

submission.to_csv("submission.csv", index=False)
print("✅ Submission file created: submission.csv")