import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# Load the dataset (make sure it's named 'adult.csv' in your folder)
df = pd.read_csv("adult.csv")

# Print columns to verify (optional)
print("Columns in dataset:", df.columns)
# Split
df_majority = df[df['income'] == '<=50K']
df_minority = df[df['income'] == '>50K']

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,
                                 n_samples=len(df_majority), 
                                 random_state=42)

# Combine
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Rename target column if needed
# Check your actual dataset — change this if necessary
if 'income' not in df.columns:
    for col in df.columns:
        if df[col].astype(str).str.contains("<=50K|>50K").any():
            df.rename(columns={col: 'income'}, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Separate features and label
X = df.drop('income', axis=1)
y = df['income']

# Label encode all categorical columns
label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

# Encode the target label
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)
label_encoders['income'] = target_encoder

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model
with open("salary_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("✅ Model and encoders trained and saved successfully.")
