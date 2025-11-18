"""
Lightweight Scam Detector using TF-IDF + Logistic Regression
This model trains much faster and works with limited memory.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json

print("=" * 70)
print(" " * 15 + "LIGHTWEIGHT SCAM DETECTOR")
print("=" * 70)

# Load processed data
print("\n📊 Loading processed datasets...")
train_df = pd.read_csv('data/processed/train.csv')
val_df = pd.read_csv('data/processed/val.csv')
test_df = pd.read_csv('data/processed/test.csv')

print(f"   Train: {len(train_df)} examples")
print(f"   Val:   {len(val_df)} examples")
print(f"   Test:  {len(test_df)} examples")

# Prepare data
X_train = train_df['text'].values
y_train = (train_df['label'] == 'scam').astype(int).values

X_val = val_df['text'].values
y_val = (val_df['label'] == 'scam').astype(int).values

X_test = test_df['text'].values
y_test = (test_df['label'] == 'scam').astype(int).values

# Create TF-IDF features
print("\n🔧 Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words='english'
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print(f"   Feature dimension: {X_train_tfidf.shape[1]}")

# Train model
print("\n🚀 Training Logistic Regression model...")
model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train_tfidf, y_train)
print("   ✅ Training complete!")

# Evaluate
print("\n📊 Evaluating model...")

# Training set
train_preds = model.predict(X_train_tfidf)
train_acc = accuracy_score(y_train, train_preds)

# Validation set  
val_preds = model.predict(X_val_tfidf)
val_acc = accuracy_score(y_val, val_preds)

# Test set
test_preds = model.predict(X_test_tfidf)
test_acc = accuracy_score(y_test, test_preds)
test_precision = precision_score(y_test, test_preds)
test_recall = recall_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds)

print(f"\n   Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   Val Accuracy:   {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"   Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")

print("\n" + "=" * 70)
print("📊 FINAL TEST SET METRICS")
print("=" * 70)
print(f"Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
print(f"Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
print(f"F1 Score:  {test_f1:.4f} ({test_f1*100:.2f}%)")
print("=" * 70)

# Check targets
print("\n🎯 Target Check:")
recall_target = test_recall >= 0.90
precision_target = test_precision >= 0.80

print(f"   Recall > 90%:    {'✅ PASS' if recall_target else '❌ FAIL'}")
print(f"   Precision > 80%: {'✅ PASS' if precision_target else '❌ FAIL'}")

# Detailed classification report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, test_preds, target_names=['Legitimate', 'Scam']))

# Confusion matrix
print("\n📈 Creating confusion matrix...")
cm = confusion_matrix(y_test, test_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Legitimate', 'Scam'],
    yticklabels=['Legitimate', 'Scam']
)
plt.title('Confusion Matrix - Lightweight Scam Detector', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()

Path('outputs').mkdir(exist_ok=True)
plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved to outputs/confusion_matrix.png")

# Save model
print("\n💾 Saving model...")
Path('models/lightweight_model').mkdir(parents=True, exist_ok=True)

with open('models/lightweight_model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('models/lightweight_model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save metrics
metrics = {
    'accuracy': float(test_acc),
    'precision': float(test_precision),
    'recall': float(test_recall),
    'f1': float(test_f1),
    'confusion_matrix': cm.tolist(),
    'model_type': 'TF-IDF + Logistic Regression',
    'feature_count': int(X_train_tfidf.shape[1]),
    'training_samples': int(len(y_train))
}

with open('outputs/evaluation_report.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"   ✅ Model saved to models/lightweight_model/")
print(f"   ✅ Metrics saved to outputs/evaluation_report.json")

# Test on sample messages
print("\n" + "=" * 70)
print("🧪 TESTING ON SAMPLE MESSAGES")
print("=" * 70)

samples = [
    "Congratulations! You've won $1,000,000! Click here to claim your prize now!",
    "Hi, I wanted to follow up on our meeting yesterday. Are you available next week?",
    "URGENT: Your account will be closed unless you verify your information immediately!",
    "Thanks for the update. I'll review the document and get back to you tomorrow.",
    "You have been selected for a FREE iPhone! Limited time offer. Act now!",
]

for i, text in enumerate(samples, 1):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    
    label = "SCAM" if prediction == 1 else "LEGIT"
    confidence = probability[prediction] * 100
    
    print(f"\n{i}. {text[:60]}...")
    print(f"   Prediction: {label} (confidence: {confidence:.1f}%)")

print("\n" + "=" * 70)
print("✅ TRAINING AND EVALUATION COMPLETE!")
print("=" * 70)
print("\n📁 Deliverables:")
print("   ✅ Trained model: models/lightweight_model/")
print("   ✅ Confusion matrix: outputs/confusion_matrix.png")
print("   ✅ Evaluation report: outputs/evaluation_report.json")
print("\n" + "=" * 70)
