Fraud Detection using Machine Learning

This project applies Machine Learning to detect fraudulent transactions in financial datasets. Using Random Forest Classifier and SMOTE for class balancing, the model learns patterns to distinguish between fraud and non-fraud cases.

Features

Data preprocessing and cleaning

Balanced dataset with SMOTE

Random Forest Classifier for classification

Evaluation with Confusion Matrix, Classification Report, ROC Curve, and AUC

Model persistence using joblib

Dataset

The project uses a fraud detection dataset (fraud.csv).
The dataset is not included in this repository due to size. To run locally, place your dataset in a data/ folder.

Results

Confusion Matrix visualization

ROC Curve with AUC score

Accuracy and classification report

(Sample outputs are available in the docs/ folder.)

Tech Stack

Python

NumPy, Pandas, Seaborn, Matplotlib

scikit-learn, imbalanced-learn (SMOTE), Joblib

Skills Strengthened

Data preprocessing and feature engineering

Handling imbalanced datasets

Supervised learning (Random Forest)

Model evaluation and visualization

How to Run

Clone this repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Install dependencies:

pip install -r requirements.txt


Add dataset in data/fraud.csv.

Run the script:

python fraud_detection.py
