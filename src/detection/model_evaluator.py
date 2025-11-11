"""
Model Evaluator for Scam Detection ML Pipeline
Evaluates trained models on test data and generates performance metrics.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json


class ModelEvaluator:
    def __init__(self, model_path="models/best_model"):
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processed_data_path = Path("data/processed")
        self.outputs_path = Path("outputs")
        self.outputs_path.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        
        print(f"🔧 Using device: {self.device}")
    
    def load_model(self):
        """
        Load the trained model and tokenizer.
        """
        print(f"📦 Loading model from {self.model_path}...")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"   ✅ Model loaded successfully")
    
    def predict_batch(self, texts, batch_size=16):
        """
        Run inference on a batch of texts.
        """
        predictions = []
        probabilities = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encodings = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)
    
    def evaluate_model(self, test_set_path="data/processed/test.csv"):
        """
        Evaluate model on test set and return predictions and labels.
        """
        print("🧪 Evaluating model on test set...")
        
        # Load test data
        test_df = pd.read_csv(test_set_path)
        texts = test_df['text'].values.tolist()
        true_labels = (test_df['label'] == 'scam').astype(int).values
        
        print(f"   Test set: {len(texts)} examples")
        
        # Get predictions
        predictions, probabilities = self.predict_batch(texts)
        
        return predictions, true_labels, probabilities, test_df
    
    def calculate_metrics(self, predictions, true_labels):
        """
        Calculate comprehensive metrics.
        """
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions),
            'recall': recall_score(true_labels, predictions),
            'f1': f1_score(true_labels, predictions)
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """
        Display metrics in a formatted way.
        """
        print("\n" + "=" * 60)
        print("📊 MODEL PERFORMANCE METRICS")
        print("=" * 60)
        print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1 Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
        print("=" * 60)
        
        # Check targets
        print("\n🎯 Target Check:")
        recall_target = metrics['recall'] >= 0.90
        precision_target = metrics['precision'] >= 0.80
        
        print(f"   Recall > 90%:    {'✅ PASS' if recall_target else '❌ FAIL'}")
        print(f"   Precision > 80%: {'✅ PASS' if precision_target else '❌ FAIL'}")
        
        if recall_target and precision_target:
            print("\n🎉 Model meets all targets!")
        else:
            print("\n⚠️  Model needs improvement to meet targets.")
    
    def create_confusion_matrix(self, predictions, true_labels, save_path="outputs/confusion_matrix.png"):
        """
        Create and save confusion matrix visualization.
        """
        print("\n📈 Creating confusion matrix...")
        
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Legitimate', 'Scam'],
            yticklabels=['Legitimate', 'Scam']
        )
        plt.title('Confusion Matrix - Scam Detection Model', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Confusion matrix saved to {save_path}")
        
        return cm
    
    def analyze_errors(self, predictions, true_labels, test_df, num_examples=5):
        """
        Analyze and display misclassified examples.
        """
        print("\n" + "=" * 60)
        print("🔍 ERROR ANALYSIS")
        print("=" * 60)
        
        # Find false positives and false negatives
        false_positives = (predictions == 1) & (true_labels == 0)
        false_negatives = (predictions == 0) & (true_labels == 1)
        
        fp_count = false_positives.sum()
        fn_count = false_negatives.sum()
        
        print(f"\nFalse Positives (legit → scam): {fp_count}")
        print(f"False Negatives (scam → legit): {fn_count}")
        
        # Show examples of false positives
        if fp_count > 0:
            print(f"\n❌ False Positive Examples (showing up to {num_examples}):")
            print("-" * 60)
            fp_indices = np.where(false_positives)[0][:num_examples]
            for i, idx in enumerate(fp_indices, 1):
                text = test_df.iloc[idx]['text']
                print(f"\n{i}. {text[:200]}...")
        
        # Show examples of false negatives
        if fn_count > 0:
            print(f"\n❌ False Negative Examples (showing up to {num_examples}):")
            print("-" * 60)
            fn_indices = np.where(false_negatives)[0][:num_examples]
            for i, idx in enumerate(fn_indices, 1):
                text = test_df.iloc[idx]['text']
                print(f"\n{i}. {text[:200]}...")
        
        print("\n" + "=" * 60)
    
    def test_edge_cases(self):
        """
        Test model on specific edge cases.
        """
        print("\n" + "=" * 60)
        print("🧪 EDGE CASE TESTING")
        print("=" * 60)
        
        edge_cases = [
            ("Short msg", "Win now!"),
            ("Very long message", "Hello " * 100 + "Please help with this matter."),
            ("With emojis", "🎉 Congratulations! You've won $1000! 💰 Click here now! 🔥"),
            ("Mixed case", "WiN A fReE iPhOnE tOdAy!!!"),
            ("Normal email", "Hi, I wanted to follow up on our meeting yesterday. Can we reschedule for next week?"),
            ("Technical jargon", "The API endpoint returned a 503 error. Please check the load balancer configuration."),
            ("Numbers only", "123456789"),
            ("Special chars", "!!!??? $$$ %%% @@@"),
        ]
        
        for name, text in edge_cases:
            predictions, probabilities = self.predict_batch([text])
            pred_label = "SCAM" if predictions[0] == 1 else "LEGIT"
            confidence = probabilities[0][predictions[0]] * 100
            
            print(f"\n{name}:")
            print(f"  Text: {text[:80]}...")
            print(f"  Prediction: {pred_label} (confidence: {confidence:.1f}%)")
        
        print("\n" + "=" * 60)
    
    def save_evaluation_report(self, metrics, confusion_mat):
        """
        Save evaluation report to JSON file.
        """
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'metrics': metrics,
            'confusion_matrix': confusion_mat.tolist(),
            'targets': {
                'recall_target': 0.90,
                'precision_target': 0.80,
                'recall_met': metrics['recall'] >= 0.90,
                'precision_met': metrics['precision'] >= 0.80
            }
        }
        
        report_path = self.outputs_path / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Evaluation report saved to {report_path}")
    
    def run_full_evaluation(self):
        """
        Run complete evaluation pipeline.
        """
        print("=" * 60)
        print("🚀 STARTING MODEL EVALUATION")
        print("=" * 60)
        
        # Load model
        self.load_model()
        
        # Evaluate on test set
        predictions, true_labels, probabilities, test_df = self.evaluate_model()
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, true_labels)
        self.print_metrics(metrics)
        
        # Create confusion matrix
        confusion_mat = self.create_confusion_matrix(predictions, true_labels)
        
        # Analyze errors
        self.analyze_errors(predictions, true_labels, test_df)
        
        # Test edge cases
        self.test_edge_cases()
        
        # Save report
        self.save_evaluation_report(metrics, confusion_mat)
        
        print("\n" + "=" * 60)
        print("✅ EVALUATION COMPLETE")
        print("=" * 60)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Model Evaluator for Scam Detection')
    parser.add_argument('--test', action='store_true', help='Run evaluation and print metrics')
    parser.add_argument('--model-path', type=str, default='models/best_model', 
                       help='Path to trained model')
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(model_path=args.model_path)
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
