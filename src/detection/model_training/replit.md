# Scam Detection ML Training Pipeline

## Project Overview
This is a complete machine learning training system for detecting scam messages. The project was created for Jack's role in the scam detection team - training the AI brain that powers the whole scam detection system.

## Purpose
Build, train, and evaluate transformer-based models (DistilBERT) to classify messages as scam or legitimate with high accuracy (>90% recall, >80% precision).

## Current State
**Status**: ✅ Initial implementation complete

The project includes:
- **Data Compilation Module** (`data_compiler.py`): Loads, cleans, balances, and splits datasets
- **Model Training Module** (`model_trainer.py`): Fine-tunes DistilBERT on scam detection
- **Model Evaluation Module** (`model_evaluator.py`): Tests models and generates comprehensive metrics
- **Main Orchestrator** (`main.py`): Runs the complete pipeline end-to-end

## Recent Changes
**2025-10-26**: Initial project setup
- Created three-module ML pipeline (compiler, trainer, evaluator)
- Generated sample scam and legitimate message datasets (40 examples each)
- Installed ML dependencies: transformers, torch, pandas, scikit-learn, matplotlib
- Implemented 80/10/10 train/validation/test data splitting
- Added confusion matrix visualization and error analysis
- Created comprehensive documentation and README

## Project Architecture

### Directory Structure
```
├── data/
│   ├── raw/              # Input CSV files (scam_*.csv, legit_*.csv)
│   └── processed/        # Cleaned train/val/test splits
├── models/
│   └── best_model/       # Trained model checkpoints
├── outputs/
│   ├── confusion_matrix.png
│   └── evaluation_report.json
├── data_compiler.py      # Dataset processing
├── model_trainer.py      # Model training
├── model_evaluator.py    # Model evaluation
└── main.py              # Pipeline orchestrator
```

### Key Technologies
- **Model**: DistilBERT (distilbert-base-uncased) - lightweight transformer
- **Framework**: PyTorch + HuggingFace Transformers
- **Data Processing**: Pandas + NumPy
- **Evaluation**: Scikit-learn metrics + Matplotlib/Seaborn visualizations

### Data Pipeline
1. Load CSV files from `data/raw/`
2. Clean text (remove duplicates, invalid entries)
3. Balance classes (equal scam/legit samples)
4. Split 80% train / 10% validation / 10% test
5. Save to `data/processed/`

### Training Pipeline
1. Load DistilBERT base model
2. Tokenize text (max 256 tokens)
3. Fine-tune with AdamW optimizer
4. Save best model based on validation loss
5. Log training metrics per epoch

### Evaluation Pipeline
1. Load trained model from checkpoint
2. Run inference on test set
3. Calculate accuracy, precision, recall, F1
4. Generate confusion matrix visualization
5. Analyze misclassified examples
6. Test edge cases (short messages, emojis, etc.)

## User Preferences
- **Code Style**: Professional, well-documented Python with type hints
- **Metrics Priority**: Recall > Precision (better to flag too much than miss real scams)
- **Model Choice**: DistilBERT for balance of speed and accuracy
- **Target Goals**: Recall >90%, Precision >80%

## How to Use

### Test Your Trained Model (Start Here!)
```bash
python test_scam_detector.py
```
Choose from 3 modes:
- **Predefined examples**: Test on 10 sample messages  
- **Your own messages**: Enter custom text to test
- **Interactive mode**: Real-time testing

### Train a New Model

**Recommended (Lightweight - Works on CPU):**
```bash
python main.py --compile-only         # Process your data
python train_simple_model.py          # Train model (fast!)
```

**Advanced (Transformer - Needs GPU for large datasets):**
```bash
python main.py --full-pipeline        # Complete pipeline
python main.py --train-only --epochs 5 --batch-size 32  # Custom training
```

### Data Processing Only
```bash
python data_compiler.py --test        # Show dataset statistics
python main.py --compile-only         # Process and split data
```

## Team Collaboration

### Deliverables for Other Team Members
- **Samya** (Deployment): Share `models/best_model/` directory with trained model + tokenizer
- **Nicole** (Dashboard): Provide `outputs/confusion_matrix.png` for visualization
- **Rohan/Oma** (Data Collection): Request real-world emails/transcripts for retraining
- **Vanya** (Management): Send weekly `outputs/evaluation_report.json` updates

## Sample Datasets
The project includes 40 scam examples and 40 legitimate examples for testing:
- `data/raw/scam_messages.csv`: Phishing emails, lottery scams, fake warnings
- `data/raw/legit_messages.csv`: Normal emails, appointments, work messages

For production use, add larger datasets (2000+ examples recommended).

## Performance Targets
- **Accuracy**: Overall correctness
- **Precision**: Of predicted scams, how many are real? (Target: >80%)
- **Recall**: Of all real scams, how many did we catch? (Target: >90%)
- **F1 Score**: Harmonic mean of precision and recall

Recall is prioritized because missing a real scam is worse than a false alarm.

## Next Steps (Stretch Goals)
1. Integrate Kaggle/HuggingFace dataset auto-download
2. Add TensorBoard for training visualization
3. Train separate models for email vs voice transcripts
4. Implement multilingual support (Hindi, Spanish)
5. Create ensemble model (voting between multiple models)
6. Set up automated weekly retraining

## Notes
- Training uses CPU by default (GPU auto-detected if available)
- First run downloads DistilBERT model (~250MB)
- Larger datasets improve model performance
- Model can be retrained as new scam examples emerge
