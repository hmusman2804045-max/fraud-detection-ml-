# First we are going to import out libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE 
import joblib
# load the dataset and read it and view first 5 rows to check
# if we have imported the correct dataset

df = pd.read_csv('fraud.csv')
print(df.head())

# Check the shape of the dataset (rows, columns)
print("Dataset shape:", df.shape)

# Check if there are any missing values
print("Missing values:\n", df.isnull().sum())

# See basic statistics of the dataset
print("\nSummary statistics:\n", df.describe())

# Check how many fraud and non-fraud transactions there are
print("\nClass distribution:\n", df['Class'].value_counts())

# X = df.drop('Class', axis=1)

X = df.drop('Class', axis=1)
y = df['Class']
# Split into training and testing sets (with stratify to maintain class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))


# Visualize the confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

joblib.dump(model, 'fraud_model.pkl')

y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print("AUC Score:", roc_auc_score(y_test, y_probs))

