import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import pickle
import re
import argparse
import os
import sys
import psutil
from urllib.parse import urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to check available memory
def check_memory_requirements(n_samples, n_features):
    estimated_memory = n_samples * n_features * 8 / (1024 ** 3)  # GB
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
    if estimated_memory > available_memory * 0.8:
        logging.warning(f"Estimated memory ({estimated_memory:.2f} GB) exceeds 80% of available memory ({available_memory:.2f} GB). Consider reducing dataset size or features.")
    return estimated_memory, available_memory

# Function to extract URL features
def extract_url_features(text):
    if not isinstance(text, str):
        return 0, 0
    urls = re.findall(r'http[s]?://(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^?\s]*)?', text)
    url_count = len(urls)
    suspicious_url = 0
    for url in urls:
        try:
            parsed = urlparse(url)
            if parsed.scheme == 'http':
                suspicious_url = 1
                break
        except ValueError as e:
            logging.warning(f"Skipping invalid URL: {url} - Error: {str(e)}")
            continue
    return url_count, suspicious_url

# Function to extract keyword features
def extract_keyword_features(text):
    if not isinstance(text, str):
        return 0
    keywords = [
        'urgent', 'verify', 'click', 'login', 'suspended', 'win', 'prize',
        'donate', 'relief', 'foundation', 'charity', 'victim', 'fund',
        'contribute', 'redeem', 'survey', 'bank', 'password', 'security',
        'action', 'required', 'confirm', 'access', 'account'
    ]
    text_lower = text.lower()
    keyword_count = 0
    for keyword in keywords:
        if keyword in ['account', 'update', 'security', 'password', 'confirm']:
            if (f'http://' in text_lower and 
                sum(k in text_lower for k in ['urgent', 'verify', 'suspended', 'click']) >= 2):
                keyword_count += (keyword in text_lower)
        else:
            keyword_count += (keyword in text_lower)
    return keyword_count

# Function to clean email text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = ' '.join(text.split())
    return text

# Function to prepare dataset
def prepare_dataset(max_samples=20000):
    dataset_path = 'combined_dataset.csv'
    fallback_path = 'CEAS_08.csv'
    
    if os.path.exists(dataset_path):
        print(f"Using combined dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
    elif os.path.exists(fallback_path):
        print(f"Warning: Combined dataset not found. Falling back to {fallback_path}")
        df = pd.read_csv(fallback_path)
    else:
        print("Error: No dataset found. Please provide combined_dataset.csv or CEAS_08.csv")
        sys.exit(1)
    
    required_columns = ['body', 'label']
    if not all(col in df.columns for col in required_columns):
        print("Error: Dataset missing required columns (body, label)")
        sys.exit(1)
    
    phishing_count = sum(df['label'] == 1)
    legit_count = sum(df['label'] == 0)
    target_per_class = min(max_samples // 2, min(phishing_count, legit_count))
    if len(df) > 2 * target_per_class:
        print(f"Dataset too large ({len(df)}). Subsampling to {2 * target_per_class} emails.")
        phishing_df = df[df['label'] == 1].sample(n=target_per_class, random_state=42)
        legit_df = df[df['label'] == 0].sample(n=target_per_class, random_state=42)
        df = pd.concat([phishing_df, legit_df])
    
    df['cleaned_body'] = df['body'].apply(clean_text)
    if 'subject' in df.columns:
        df['cleaned_subject'] = df['subject'].apply(clean_text)
        df['text'] = df['cleaned_body'] + ' ' + df['cleaned_subject'].fillna('')
    else:
        df['text'] = df['cleaned_body']
    
    try:
        df['url_count'], df['suspicious_url'] = zip(*df['body'].apply(extract_url_features))
        df['keyword_count'] = df['body'].apply(extract_keyword_features)
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        sys.exit(1)
    
    df = df[df['text'] != '']
    
    if not df['label'].isin([0, 1]).all():
        print("Error: Labels must be binary (0 or 1)")
        sys.exit(1)
    
    suspicious_legit = df[(df['label'] == 0) & (df['suspicious_url'] == 1)].shape[0]
    if suspicious_legit > 0:
        logging.warning(f"Found {suspicious_legit} legitimate emails with suspicious URLs. Check dataset labels.")
    
    suspicious_phishing = df[(df['label'] == 1) & (df['suspicious_url'] == 1)].shape[0]
    keyword_phishing = df[(df['label'] == 1) & (df['keyword_count'] >= 3)].shape[0]
    print(f"Phishing emails with suspicious URLs: {suspicious_phishing} ({100 * suspicious_phishing / sum(df['label'] == 1):.2f}%)")
    print(f"Phishing emails with 3+ keywords: {keyword_phishing} ({100 * keyword_phishing / sum(df['label'] == 1):.2f}%)")
    
    print(f"Dataset size: {len(df)} emails")
    print(f"Phishing emails: {sum(df['label'] == 1)} ({100 * sum(df['label'] == 1) / len(df):.2f}%)")
    print(f"Legitimate emails: {sum(df['label'] == 0)} ({100 * sum(df['label'] == 0) / len(df):.2f}%)")
    
    return df

# Function to train model
def train_model():
    df = prepare_dataset()
    
    stop_words = [
        'thank', 'order', 'purchase', 'confirmed', 'track', 'contact', 'please',
        'dear', 'team', 'customer', 'service', 'best', 'regards', 'sincerely',
        'visit', 'website', 'support', 'items', 'business', 'days', 'reset',
        'registration', 'renew', 'details', 'member', 'password', 'account',
        'user', 'successfully', 'change', 'request', 'view', 'annual', 'valued',
        'security', 'login', 'new', 'conference', 'attendee', 'subscription', 'payment',
        'example', 'inc', 'newsletter', 'preferences', 'updates'
    ]
    vectorizer = TfidfVectorizer(max_features=100, min_df=5, stop_words=stop_words)
    X_text = vectorizer.fit_transform(df['text'])
    
    scaler = StandardScaler()
    metadata_features = scaler.fit_transform(df[['url_count', 'suspicious_url', 'keyword_count']].values)
    
    metadata_features *= 4
    
    X = hstack([X_text, metadata_features])
    y = df['label']
    
    estimated_memory, available_memory = check_memory_requirements(X.shape[0], X.shape[1])
    print(f"Estimated memory: {estimated_memory:.2f} GB, Available memory: {available_memory:.2f} GB")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scale_pos_weight = 4 * sum(y == 0) / sum(y == 1) if sum(y == 1) > 0 else 1
    model = XGBClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        reg_lambda=1.0,
        reg_alpha=0.5
    )
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"5-Fold Cross-Validation Accuracy: {cv_scores.mean():.2f} (±{cv_scores.std() * 2:.2f})")
    
    model = XGBClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        reg_lambda=1.0,
        reg_alpha=0.5,
        early_stopping_rounds=10
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Set Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    feature_importance = model.feature_importances_[-3:]
    print("\nFeature Importance for Metadata:")
    print(f"URL Count: {feature_importance[0]:.4f}")
    print(f"Suspicious URL: {feature_importance[1]:.4f}")
    print(f"Keyword Count: {feature_importance[2]:.4f}")
    
    with open('phishing_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, vectorizer, scaler

# Function to predict email
def predict_email(email_text, model, vectorizer, scaler, debug=False, threshold=0.95):
    cleaned_text = clean_text(email_text)
    if not cleaned_text:
        return "Error: Empty or invalid email content."
    
    url_count, suspicious_url = extract_url_features(email_text)
    keyword_count = extract_keyword_features(email_text)
    
    X_text = vectorizer.transform([cleaned_text])
    
    metadata_features = scaler.transform([[url_count, suspicious_url, keyword_count]]) * 4
    
    X = hstack([X_text, metadata_features])
    
    prob = model.predict_proba(X)[0][1]
    prediction = 1 if prob >= threshold else 0
    result = "PHISHING" if prediction == 1 else "LEGIT"
    
    if debug:
        print(f"Debug Info:")
        print(f"Text (first 100 chars): {cleaned_text[:100]}...")
        print(f"URL Count: {url_count}")
        print(f"Suspicious URL: {suspicious_url}")
        print(f"Keyword Count: {keyword_count}")
        print(f"Phishing Probability: {prob:.4f}")
    
    return result

# CLI Interface
def main():
    parser = argparse.ArgumentParser(
        description="Phishing Email Detector: Classify emails as PHISHING or LEGIT",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the model using combined_dataset.csv or CEAS_08.csv'
    )
    parser.add_argument(
        '--predict',
        type=str,
        help='Predict if an email is phishing\nExample: python phishing_detector.py --predict \"Subject: Urgent...\"'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Predict email from a text file\nExample: python phishing_detector.py --file email.txt'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output for predictions'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.95,
        help='Prediction threshold for phishing probability (0.0 to 1.0, default: 0.95)'
    )
    
    args = parser.parse_args()
    
    model_path = 'phishing_model.pkl'
    vectorizer_path = 'vectorizer.pkl'
    scaler_path = 'scaler.pkl'
    
    if args.train or not (os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(scaler_path)):
        print("Training model...")
        try:
            model, vectorizer, scaler = train_model()
        except Exception as e:
            print(f"Error during training: {str(e)}")
            sys.exit(1)
    else:
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        except Exception as e:
            print(f"Error loading model, vectorizer, or scaler: {str(e)}")
            print("Please run with --train to create model files.")
            sys.exit(1)
    
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                email_text = f.read()
            result = predict_email(email_text, model, vectorizer, scaler, debug=args.debug, threshold=args.threshold)
            print(f"Prediction: {result}")
        except FileNotFoundError:
            print(f"Error: File {args.file} not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            sys.exit(1)
    elif args.predict:
        result = predict_email(args.predict, model, vectorizer, scaler, debug=args.debug, threshold=args.threshold)
        print(f"Prediction: {result}")
    else:
        print("=====================================")
        print("Phishing Email Detector")
        print("=====================================")
        print("Instructions:")
        print("- Paste the full email content (subject and body).")
        print("- Press Enter twice to submit.")
        print("- Type 'quit' to exit.")
        print("=====================================\n")
        
        while True:
            print("Enter email content:")
            lines = []
            empty_line_count = 0
            
            while True:
                try:
                    line = input()
                except EOFError:
                    print("\nError: Input interrupted. Exiting...")
                    sys.exit(1)
                
                if line.lower() == 'quit':
                    print("Exiting...")
                    sys.exit(0)
                
                if line == "":
                    empty_line_count += 1
                    if empty_line_count >= 2 and lines:
                        break
                else:
                    empty_line_count = 0
                    lines.append(line)
                
                if len(lines) > 1000:
                    print("\nError: Input too long. Please submit a shorter email.")
                    lines = []
                    break
            
            if lines:
                email_text = " ".join(lines)
                result = predict_email(email_text, model, vectorizer, scaler, debug=args.debug, threshold=args.threshold)
                print(f"\nPrediction: {result}\n")
            else:
                print("\nNo input provided. Please paste an email or type 'quit' to exit.\n")
            
            print("=====================================")
            print("Paste another email or type 'quit' to exit:")

if __name__ == "__main__":
    main()