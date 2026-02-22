import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("data/Purchase_Logistics.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# 2. Basic preprocessing
# Change target column name if needed
X = df.drop("Purchased", axis=1)
y = df["Purchased"]

# 3. Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model training
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 6. Evaluation
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Save model and scaler
joblib.dump(model, "artifacts/model.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")

print("Model and scaler saved successfully!")