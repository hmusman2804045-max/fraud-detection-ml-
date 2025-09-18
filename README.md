# Fraud Detection using Machine Learning  

This project applies Machine Learning to detect fraudulent transactions in financial datasets. Using a Random Forest Classifier and SMOTE for class balancing, the model learns patterns to distinguish between fraud and non-fraud cases.  

## Features  
- Data preprocessing and cleaning  
- Balanced dataset with SMOTE  
- Random Forest Classifier for classification  
- Evaluation with Confusion Matrix, Classification Report, ROC Curve, and AUC  
- Model persistence using joblib  

## Dataset  
The project uses a fraud detection dataset (`fraud.csv`).  
The dataset is not included in this repository due to size. To run locally, place your dataset in a `data/` folder.  

## Results  
The model was evaluated on the test set and achieved strong performance:  
- Confusion Matrix visualization showing classification performance  
- ROC Curve with AUC score for fraud detection capability  
- Accuracy and classification report for detailed metrics  

Sample outputs:  
- `docs/confusion_matrix.png`  
- `docs/roc_curve.png`  

## Installation  

```bash
# Clone the repository
git clone https://github.com/your-username/fraud-detection-ml.git
cd fraud-detection-ml

# Install dependencies
pip install -r requirements.txt

# Place the dataset in the data/ folder
mkdir data
# Copy your fraud.csv file into the data folder

# Run the script
python fraud_detection.py
