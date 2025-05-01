# Phishing Email Detector

## Overview
The **Phishing Email Detector** is a Python-based command-line tool designed to classify emails as `PHISHING` or `LEGIT`. This project leverages machine learning to achieve high accuracy in detecting phishing emails.

The tool processes email text (subject and body), extracts text and metadata features, and uses an XGBoost classifier to make predictions. It’s user-friendly, supports batch testing, and includes debug output for transparency.

## Project Flow
Here’s how the project was developed and how it works:

### 1. Problem Definition
Phishing emails are a major cybersecurity threat, tricking users into sharing sensitive information. The goal was to build a detector that:
- Achieves **near-perfect legitimate recall** (0-1 false positives) to avoid flagging legitimate emails.
- Maintains high phishing detection accuracy.
- Provides a simple CLI interface for training and prediction.

### 2. Dataset Preparation
- **Source**: Used `combined_dataset.csv` (containing datasets - CEAS_08.csv, NigerianFraud, SpamAssasin, Nazario) (or only `CEAS_08.csv` as fallback), containing email bodies, subjects, and binary labels (`0=LEGIT`, `1=PHISHING`).
- **Preprocessing**:
  - Subsampled up to 20,000 emails (10,000 per class) for efficiency.
  - Cleaned text by converting to lowercase and removing extra spaces.
  - Combined subject and body into a single `text` field.
- **Feature Extraction**:
  - **Text Features**: Used `TfidfVectorizer` (max 100 features, min document frequency 5) with stop words (e.g., "thank," "order") to remove noise.
  - **Metadata Features**:
    - `url_count`: Number of URLs in the email.
    - `suspicious_url`: 1 if any URL uses HTTP (not HTTPS), 0 otherwise.
    - `keyword_count`: Count of phishing-related keywords (e.g., "urgent," "verify") with conditional logic for ambiguous terms (e.g., "account" only counts with HTTP URLs and 2+ strong phishing keywords).

### 3. Model Training
- **Algorithm**: XGBoost classifier, chosen for its performance on imbalanced data.
- **Parameters**:
  - `n_estimators=50`, `learning_rate=0.1`, `max_depth=6`.
  - `scale_pos_weight=4` to prioritize legitimate emails.
  - `reg_lambda=1.0`, `reg_alpha=0.5` for regularization.
  - Early stopping after 10 rounds.
- **Feature Engineering**:
  - Combined TF-IDF text features with scaled metadata features (weighted by *4).
  - Used `StandardScaler` to normalize metadata.
- **Evaluation**:
  - 5-fold cross-validation to assess accuracy.
  - Test set evaluation with accuracy, precision, recall, F1-score, and confusion matrix.
  - Focused on maximizing legitimate recall to meet the ~0.999999 target.

### 4. Prediction
- **Input**: Email text (subject + body) via CLI, file, or direct input.
- **Process**:
  - Clean text and extract features (`url_count`, `suspicious_url`, `keyword_count`).
  - Transform text using the trained `TfidfVectorizer`.
  - Combine with scaled metadata and predict using XGBoost.
  - Apply a high threshold (0.95) to ensure only confident phishing predictions are flagged.
- **Output**: `PHISHING` or `LEGIT`, with optional debug info (text snippet, feature values, phishing probability).

### 5. Testing and Validation
- **Test Cases**:
  - Tested 13 legitimate emails (e.g., order confirmations, subscription renewals) to ensure all are classified as `LEGIT` with low phishing probabilities (<0.95).
  - Tested 3 phishing emails (e.g., account suspension scams) to confirm `PHISHING` classification with high probabilities (>0.95).
- **Performance**:
  - Achieved near-perfect legitimate recall (~1.0, 0-1 false positives).
  - High phishing recall (~0.95) and overall accuracy (~0.95).
  - Confusion matrix showed minimal false positives.

### 6. Deployment
- Saved trained model (`phishing_model.pkl`), vectorizer (`vectorizer.pkl`), and scaler (`scaler.pkl`) for reuse.
- Provided a CLI interface for easy training and prediction.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/saniyanz/bytebash_hack.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn xgboost scipy psutil
   ```
3. **Prepare Dataset**:
   - Place `combined_dataset.csv` (or `CEAS_08.csv`) in the project directory.
   - Ensure it has `body` and `label` columns (optional: `subject`).

## Usage
### Training the Model
```bash
python phishing_detector.py --train
```
- Loads and preprocesses the dataset.
- Trains the XGBoost model and saves it.
- Outputs cross-validation accuracy, test set metrics, and feature importance.

### Predicting a Single Email
```bash
python phishing_detector.py --predict "Subject: Your Order Confirmed Thank you for your purchase..."
```
- Outputs `PHISHING` or `LEGIT`.
- Add `--debug` for feature values and phishing probability:
  ```bash
  python phishing_detector.py --predict "..." --debug
  ```

### Predicting from a File
```bash
python phishing_detector.py --file email.txt --debug
```
- Reads `email.txt` and outputs the prediction.

### Interactive Mode
```bash
python phishing_detector.py
```
- Paste email content, press Enter twice to submit, or type `quit` to exit.

### Adjusting Threshold
```bash
python phishing_detector.py --predict "..." --threshold 0.9
```
- Lower threshold (e.g., 0.9) increases phishing Rosinette plots are not supported in this version of the software.
