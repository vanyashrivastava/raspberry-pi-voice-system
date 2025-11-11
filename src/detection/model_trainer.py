"""
Model Trainer for Scam Detection ML Pipeline
Fine-tunes DistilBERT on scam/legitimate message classification.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from datetime import datetime


class ScamDataset(Dataset):
    """Custom Dataset for scam detection."""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = 1 if self.labels[idx] == 'scam' else 0
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ModelTrainer:
    def __init__(self, model_name='distilbert-base-uncased', random_seed=42):
        self.model_name = model_name
        self.random_seed = random_seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        self.processed_data_path = Path("data/processed")
        self.models_path = Path("models")
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.training_history = []
        
        print(f"🔧 Using device: {self.device}")
    
    def load_base_model(self):
        """
        Load pre-trained DistilBERT model and tokenizer from HuggingFace.
        """
        print(f"📦 Loading base model: {self.model_name}...")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        
        self.model.to(self.device)
        
        print(f"   ✅ Model loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_training_data(self, batch_size=16, max_length=256):
        """
        Load processed data and create DataLoaders.
        """
        print("📊 Preparing training data...")
        
        # Load datasets
        train_df = pd.read_csv(self.processed_data_path / "train.csv")
        val_df = pd.read_csv(self.processed_data_path / "val.csv")
        
        print(f"   Train: {len(train_df)} examples")
        print(f"   Val: {len(val_df)} examples")
        
        # Create datasets
        train_dataset = ScamDataset(
            train_df['text'].values,
            train_df['label'].values,
            self.tokenizer,
            max_length
        )
        
        val_dataset = ScamDataset(
            val_df['text'].values,
            val_df['label'].values,
            self.tokenizer,
            max_length
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        print(f"   Batch size: {batch_size}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_predictions/total_predictions:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """
        Validate the model.
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train_model(self, epochs=3, batch_size=16, learning_rate=2e-5):
        """
        Full training loop with validation.
        """
        print("=" * 60)
        print("🚀 STARTING MODEL TRAINING")
        print("=" * 60)
        
        # Prepare data
        train_loader, val_loader = self.prepare_training_data(batch_size)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        print(f"\n⚙️  Training configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Total steps: {total_steps}")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }
            self.training_history.append(metrics)
            self.log_training_metrics(metrics)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"\n💾 New best model! Saving checkpoint...")
                self.save_model("best_model")
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE")
        print("=" * 60)
        
        return self.training_history
    
    def save_model(self, model_name="scam_detector"):
        """
        Save model and tokenizer to disk.
        """
        save_path = self.models_path / model_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save training history
        history_path = save_path / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"   ✅ Model saved to {save_path}/")
    
    def log_training_metrics(self, metrics):
        """
        Log training metrics to console.
        """
        print(f"\n📊 Epoch {metrics['epoch']} Results:")
        print(f"   Train Loss: {metrics['train_loss']:.4f} | Train Acc: {metrics['train_accuracy']:.4f}")
        print(f"   Val Loss:   {metrics['val_loss']:.4f} | Val Acc:   {metrics['val_accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Model Trainer for Scam Detection')
    parser.add_argument('--test', action='store_true', help='Run quick 1-epoch test')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    trainer.load_base_model()
    
    epochs = 1 if args.test else args.epochs
    
    trainer.train_model(
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()
