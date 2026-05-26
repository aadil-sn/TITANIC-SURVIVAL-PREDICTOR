
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop cabin column
if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)

# Encode categorical values
encoder = LabelEncoder()

df['Sex'] = encoder.fit_transform(df['Sex'])
df['Embarked'] = encoder.fit_transform(df['Embarked'])

# Drop unwanted columns
drop_cols = ['PassengerId', 'Name', 'Ticket']

for col in drop_cols:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "titanic_model.pkl")

# Accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Model saved as titanic_model.pkl")
