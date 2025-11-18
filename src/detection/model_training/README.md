# Scam Detection ML Pipeline

An end-to-end machine learning pipeline for detecting scam messages using fine-tuned DistilBERT transformer models.

## 🎯 Project Overview

This pipeline trains an AI model to classify messages as either scam or legitimate. It achieves high accuracy by:
- Using state-of-the-art transformer models (DistilBERT)
- Implementing careful data cleaning and balancing
- Prioritizing recall (catching scams) over precision (avoiding false alarms)

**Target Metrics:**
- Recall > 90% (catch most scams)
- Precision > 80% (minimize false alarms)

## 📁 Project Structure

```
├── data/
│   ├── raw/              # Raw datasets (scam_*.csv, legit_*.csv)
│   └── processed/        # Cleaned and split datasets
├── models/
│   └── best_model/       # Trained model checkpoint
├── outputs/
│   ├── confusion_matrix.png
│   └── evaluation_report.json
├── data_compiler.py      # Data loading and preprocessing
├── model_trainer.py      # Model training pipeline
├── model_evaluator.py    # Model evaluation and metrics
└── main.py              # Complete pipeline orchestrator
```

## 🚀 Quick Start

### Test Your Trained Model

```bash
python test_scam_detector.py
```

Choose from 3 test modes:
1. **Predefined examples** - Test on 10 sample messages
2. **Your own messages** - Enter custom messages to test
3. **Interactive mode** - Real-time testing (type messages one by one)

### Train a New Model

```bash
# Lightweight model (works on any dataset size)
python main.py --compile-only
python train_simple_model.py

# Or use the transformer model (requires GPU for large datasets)
python main.py --full-pipeline
```

### Individual Steps

```bash
# Data compilation only
python main.py --compile-only

# Training only (with custom epochs)
python main.py --train-only --epochs 5

# Evaluation only
python main.py --evaluate-only
```

### Test Mode

```bash
# Quick test with 1 epoch
python data_compiler.py --test
python model_trainer.py --test
python model_evaluator.py --test
```

## 📊 Input Data Format

Place CSV files in `data/raw/`:
- **Scam data**: Files containing `scam` in filename (e.g., `scam_messages.csv`)
- **Legitimate data**: Files containing `legit` in filename (e.g., `legit_messages.csv`)

Each CSV should have a column with text content (automatically detected).

## 📈 Output Files

After running the pipeline:

1. **Trained Model**: `models/best_model/` 
   - Model weights and tokenizer
   - Ready for deployment by Samya

2. **Confusion Matrix**: `outputs/confusion_matrix.png`
   - Visual performance analysis
   - Share with Nicole for dashboard

3. **Evaluation Report**: `outputs/evaluation_report.json`
   - Detailed metrics in JSON format
   - Includes accuracy, precision, recall, F1 score

## 🔧 Configuration

### Training Parameters

```bash
python main.py --full-pipeline --epochs 5 --batch-size 32
```

- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Training batch size (default: 16)

### Model Selection

Edit `model_trainer.py` to change the base model:

```python
trainer = ModelTrainer(model_name='distilbert-base-uncased')
```

Other options: `bert-base-uncased`, `roberta-base`

## 📋 Requirements

- Python 3.11+
- PyTorch 2.0+
- HuggingFace Transformers
- See `requirements.txt` for complete list

## 🎓 How It Works

### 1. Data Compilation (`data_compiler.py`)
- Loads scam and legitimate messages from CSV files
- Removes duplicates and cleans text
- Balances classes (equal scam/legit samples)
- Splits into train (80%), validation (10%), test (10%)

### 2. Model Training (`model_trainer.py`)
- Fine-tunes DistilBERT on the classification task
- Uses AdamW optimizer with learning rate warmup
- Saves best model based on validation loss
- Tracks training metrics per epoch

### 3. Model Evaluation (`model_evaluator.py`)
- Evaluates on held-out test set
- Calculates accuracy, precision, recall, F1
- Generates confusion matrix visualization
- Analyzes false positives and false negatives
- Tests edge cases (short messages, emojis, etc.)

## 🤝 Team Collaboration

- **Samya**: Load `models/best_model/` for deployment
- **Rohan/Oma**: Provide real-world emails/transcripts for retraining
- **Nicole**: Use `outputs/confusion_matrix.png` for dashboard
- **Vanya**: Review weekly performance reports

## 🎯 Stretch Goals

- [ ] Train separate models for email vs voice transcripts
- [ ] Add multilingual support (Hindi, Spanish)
- [ ] Implement auto-retraining with new data
- [ ] Create ensemble model (multiple model voting)

## 📝 Notes

- The model prioritizes recall to minimize missed scams
- Training uses CPU by default (GPU detected automatically)
- Dataset size affects training time and performance
- Start with 2000+ examples for best results

## 🐛 Troubleshooting

**ImportError**: Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

**Out of Memory**: Reduce batch size:
```bash
python main.py --batch-size 8
```

**Poor Performance**: Check dataset quality and balance:
```bash
python data_compiler.py --test
```
