import pandas as pd
import numpy as np

np.random.seed(42)

# Generate synthetic dataset for telecom churn
n_samples = 1000

data = {
    'tenure': np.random.randint(1, 72, n_samples),
    'MonthlyCharges': np.random.uniform(20.0, 120.0, n_samples),
    'TotalCharges': np.random.uniform(20.0, 8000.0, n_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.3, 0.5, 0.2]),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2])
}

df = pd.DataFrame(data)

# Logic to generate Churn label
# Higher churn if Month-to-month, Fiber optic, No TechSupport, higher charges, lower tenure
prob = np.zeros(n_samples)
prob += (df['Contract'] == 'Month-to-month') * 0.3
prob += (df['InternetService'] == 'Fiber optic') * 0.2
prob += (df['TechSupport'] == 'No') * 0.1
prob += (df['MonthlyCharges'] > 80) * 0.1
prob -= (df['tenure'] > 24) * 0.2

# Add some random noise
prob += np.random.normal(0, 0.1, n_samples)
# Sigmoid function roughly
churn_prob = 1 / (1 + np.exp(-prob * 5)) 

df['Churn'] = np.where(churn_prob > 0.5, 1, 0)

df.to_csv('dataset.csv', index=False)
print("Synthetic dataset generated and saved to dataset.csv")
