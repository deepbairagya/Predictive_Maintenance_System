import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
data = pd.read_csv('dataset.csv')

# Drop non-informative columns
data_cleaned = data.drop(columns=['UDI', 'Product ID'])

# Separate features and target variable
X = data_cleaned.drop(columns=['Machine failure'])
y = data_cleaned['Machine failure']

# Define categorical and numerical columns
categorical_cols = ['Type']
numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                  'Torque [Nm]', 'Tool wear [min]']

# Preprocessing pipelines for numerical and categorical data
numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())  # Normalize numerical features
])

categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(drop='first'))  # One-hot encode categorical features
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# Addressing imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Model training: Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and preprocessor
joblib.dump(model, 'model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

# Flask application for REST API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = pd.DataFrame(data)
    
    # Load preprocessor and model
    preprocessor = joblib.load('preprocessor.pkl')
    model = joblib.load('model.pkl')
    
    # Preprocess and predict
    processed_features = preprocessor.transform(features)
    predictions = model.predict(processed_features)
    return jsonify({'predictions': predictions.tolist()})

@app.route('/monitor', methods=['GET'])
def monitor():
    return jsonify({
        'status': 'API is running',
        'model_accuracy': accuracy_score(y_test, model.predict(X_test))
    })

if __name__ == '__main__':
    app.run(debug=True)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Feature importance
feature_importances = model.feature_importances_
features = preprocessor.get_feature_names_out()
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features, palette="viridis")
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
