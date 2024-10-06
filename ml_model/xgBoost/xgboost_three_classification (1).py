# -*- coding: utf-8 -*-
"""xgBoost_three_classification.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1F6TnIj-mxP91METXznGsF0pJB47nfK_Y
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
from sklearn.utils.class_weight import compute_class_weight

# Upload the dataset
# uploaded = files.upload()

# Load the dataset
df = pd.read_csv('cleaned_test_data.csv')

# Display the first few rows and basic information about the dataset
print(df.head())
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Handle missing values (if any)
df = df.dropna()

# Function to map original severity to new categories
def map_severity(severity):
    if severity in ['No Apparent Injury']:
        return 0  # No/Minor Injury
    elif severity in ['Suspected Minor Injury', 'Possible Injury']:
        return 1  # Moderate Injury
    else:  # 'Suspected Serious Injury', 'Fatal Injury (Killed)'
        return 2  # Severe/Fatal Injury


# Filter out unwanted categories and create a new DataFrame
df_filtered = df[~df['severity'].isin(['Unknown', 'Died Prior to Crash'])].copy()

# Apply the mapping to create a new severity column
df_filtered['severity_grouped'] = df_filtered['severity'].map(map_severity)

# Identify categorical columns
categorical_columns = ['first_harmful_event', 'units', 'crash_type', 'causal_unit_action',
                       'at_intersection', 'junction', 'manner_of_collision', 'weather', 'area']

# Create a dictionary to store label encoders
label_encoders = {}

# Encode all categorical columns
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df_filtered[col] = label_encoders[col].fit_transform(df_filtered[col])

# Prepare features and target variable
X = df_filtered.drop(['crash_number', 'severity', 'severity_grouped', 'fatal', 'minor', 'serioues'], axis=1)
y = df_filtered['severity_grouped']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print unique values in y_train to confirm
print("\nUnique values in y_train:")
print(np.unique(y_train))

# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

print("Class weights:")
print(class_weight_dict)

# Create and train the XGBoost model with class weights
model = XGBClassifier(
    random_state=42,
    scale_pos_weight=class_weight_dict[1] / class_weight_dict[0],  # Adjust weight for positive class
    max_delta_step=1,  # Can help with class imbalance
    min_child_weight=5  # Increase to be more conservative with imbalanced classes
)

# Convert class weights to sample weights
sample_weights = np.array([class_weight_dict[y] for y in y_train])


# Create and train the XGBoost model
model.fit(X_train, y_train, sample_weight=sample_weights)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot feature importance
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance Score")
plt.tight_layout()
plt.show()

# Analyze the relationship between top features and crash severity
top_features = feat_importances.nlargest(5).index.tolist()

for feature in top_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_filtered[feature], y=df_filtered['severity_grouped'])
    plt.title(f"Relationship between {feature} and Crash Severity")
    plt.xlabel(feature)
    plt.ylabel("Severity (0: No/Minor, 1: Moderate, 2: Severe/Fatal)")
    plt.tight_layout()
    plt.show()

# New severity mapping for interpretation
new_severity_mapping = {
    0: "No/Minor Injury",
    1: "Moderate Injury",
    2: "Severe/Fatal Injury"
}

# Print insights
print("Insights:")
print("1. The top contributing factors to car crash severity in Alaska are:")
for i, feature in enumerate(feat_importances.nlargest(5).index, 1):
    print(f"   {i}. {feature}")

print("\n2. The model's performance in predicting crash severity can be seen in the classification report above.")
print("\n3. The confusion matrix shows the model's prediction accuracy for different severity levels.")
print("\n4. The box plots illustrate the relationship between top features and crash severity.")
print("\n5. Further analysis may be needed to understand the specific impact of each factor on crash severity.")
print("\n6. The severity levels have been grouped as follows:")
for code, severity in new_severity_mapping.items():
    print(f"   {code}: {severity}")