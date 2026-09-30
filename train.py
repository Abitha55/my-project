import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
print("Loading data...")
df = pd.read_csv('dataset.csv')

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Define categorical and numerical columns
categorical_cols = ['Contract', 'InternetService', 'TechSupport']
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])

# Create the pipeline with preprocessor and logistic regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model.fit(X_train, y_train)

# Evaluate model
score = model.score(X_test, y_test)
print(f"Model accuracy on test set: {score:.4f}")

# Save the model
print("Saving model to model/model.pkl...")
joblib.dump(model, 'model/model.pkl')
print("Model saved successfully.")
