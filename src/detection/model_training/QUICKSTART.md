# Quick Start Guide

## ✅ Your Model is Already Trained!

Your scam detection model achieved:
- **93.02% Accuracy**
- **95.38% Precision**  
- **91.18% Recall** ✅ (catches 91% of scams!)

## 🚀 Test It Right Now

Run this command:
```bash
python test_scam_detector.py
```

Then choose an option:
- **Option 1**: Test on 10 sample messages (instant results)
- **Option 2**: Enter your own messages to test
- **Option 3**: Interactive mode (type and test in real-time)

## 📁 Your Trained Model

Located in: `models/lightweight_model/`
- `vectorizer.pkl` - Text processor
- `model.pkl` - Trained classifier

## 📊 View Results

- **Confusion Matrix**: `outputs/confusion_matrix.png`
- **Metrics Report**: `outputs/evaluation_report.json`
- **Test Dataset**: `data/processed/test.csv` (129 messages)

## 🔄 Retrain with New Data

1. Add CSV files to `data/raw/`:
   - Files with "scam" in name = scam examples
   - Files with "legit" in name = legitimate examples

2. Run:
   ```bash
   python main.py --compile-only
   python train_simple_model.py
   ```

That's it! Your new model is ready in ~30 seconds.

## 💡 Tips

- The model works best with 500+ examples
- Balance your dataset (equal scam/legit messages)
- Text messages, emails, or any short text works great
- For very large datasets (10,000+), consider using Google Colab with GPU

## 🎯 Model Performance

Your current model was trained on **5,572 real spam/ham messages** and achieves professional-grade accuracy!
