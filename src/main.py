import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load and prep data
print("Loading and preparing penguin data...")
penguins = sns.load_dataset("penguins")
penguins.dropna(inplace=True)

# One-hot encode for the model
df = pd.get_dummies(penguins, columns=['island', 'sex'], drop_first=True)
X = df.drop("species", axis=1)
y = df["species"]

# Save column order for the app
model_columns = X.columns.tolist()
joblib.dump(model_columns, 'model/model_columns.pkl')

# Split, train, and evaluate
print("Training Random Forest model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(model.predict(X_test), y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the final model
print("Saving model to model/penguin_model.pkl")
joblib.dump(model, "model/penguin_model.pkl")

print("Training complete.")
