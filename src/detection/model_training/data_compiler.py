"""
Data Compiler for Scam Detection ML Pipeline
Handles data loading, cleaning, balancing, and splitting for model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import re
import argparse


class DataCompiler:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.raw_data_path = Path("data/raw")
        self.processed_data_path = Path("data/processed")
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
    
    def download_datasets(self):
        """
        Placeholder for downloading datasets from Kaggle or HuggingFace.
        For now, assumes datasets are manually placed in data/raw/
        """
        print("📥 Checking for datasets in data/raw/...")
        scam_files = list(self.raw_data_path.glob("*scam*.csv"))
        legit_files = list(self.raw_data_path.glob("*legit*.csv"))
        
        print(f"   Found {len(scam_files)} scam dataset(s)")
        print(f"   Found {len(legit_files)} legitimate dataset(s)")
        
        if not scam_files or not legit_files:
            print("⚠️  Warning: Missing datasets. Place CSV files in data/raw/")
            print("   Expected: scam_*.csv and legit_*.csv files")
        
        return scam_files, legit_files
    
    def load_scam_data(self, source) -> pd.DataFrame:
        """
        Load scam examples from a CSV file.
        Returns: DataFrame with 'text' and 'label' columns
        """
        print(f"📖 Loading scam data from {source}...")
        df = pd.read_csv(source)
        
        # Try to identify the text column
        text_col = self._identify_text_column(df)
        
        data = pd.DataFrame({
            'text': df[text_col],
            'label': 'scam'
        })
        
        print(f"   Loaded {len(data)} scam examples")
        return data
    
    def load_legitimate_data(self, source) -> pd.DataFrame:
        """
        Load legitimate (non-scam) examples from a CSV file.
        Returns: DataFrame with 'text' and 'label' columns
        """
        print(f"📖 Loading legitimate data from {source}...")
        df = pd.read_csv(source)
        
        # Try to identify the text column
        text_col = self._identify_text_column(df)
        
        data = pd.DataFrame({
            'text': df[text_col],
            'label': 'legit'
        })
        
        print(f"   Loaded {len(data)} legitimate examples")
        return data
    
    def _identify_text_column(self, df: pd.DataFrame) -> str:
        """
        Identify which column contains the text data.
        """
        possible_names = ['text', 'message', 'content', 'body', 'email', 'subject']
        
        for col in df.columns:
            if col.lower() in possible_names:
                return col
        
        # Default to first column if no match
        return df.columns[0]
    
    def clean_and_format(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the dataset:
        - Remove duplicates
        - Handle missing values
        - Standardize encoding
        - Remove very short/long messages
        """
        print("🧹 Cleaning and formatting data...")
        
        original_size = len(data)
        
        # Remove rows with missing text
        data = data.dropna(subset=['text'])
        
        # Convert text to string and strip whitespace
        data['text'] = data['text'].astype(str).str.strip()
        
        # Remove duplicates
        data = data.drop_duplicates(subset=['text'])
        
        # Remove very short messages (< 10 chars) and very long ones (> 5000 chars)
        data = data[data['text'].str.len() >= 10]
        data = data[data['text'].str.len() <= 5000]
        
        # Basic text cleaning
        data['text'] = data['text'].apply(self._clean_text)
        
        # Reset index
        data = data.reset_index(drop=True)
        
        removed = original_size - len(data)
        print(f"   Removed {removed} rows ({removed/original_size*100:.1f}%)")
        print(f"   Clean dataset: {len(data)} examples")
        
        return data
    
    def _clean_text(self, text: str) -> str:
        """
        Basic text cleaning operations.
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
        
        return text.strip()
    
    def balance_classes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the dataset to have equal scam and legitimate examples.
        Uses undersampling of the majority class.
        """
        print("⚖️  Balancing classes...")
        
        scam_count = len(data[data['label'] == 'scam'])
        legit_count = len(data[data['label'] == 'legit'])
        
        print(f"   Before: {scam_count} scam, {legit_count} legit")
        
        # Determine minority class size
        min_size = min(scam_count, legit_count)
        
        # Sample equal amounts from each class
        scam_data = data[data['label'] == 'scam'].sample(n=min_size, random_state=self.random_seed)
        legit_data = data[data['label'] == 'legit'].sample(n=min_size, random_state=self.random_seed)
        
        # Combine and shuffle
        balanced_data = pd.concat([scam_data, legit_data], ignore_index=True)
        balanced_data = balanced_data.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        print(f"   After: {len(balanced_data)} total ({min_size} per class)")
        
        return balanced_data
    
    def split_train_val_test(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train (80%), validation (10%), and test (10%) sets.
        """
        print("✂️  Splitting into train/val/test sets...")
        
        # Shuffle data
        data = data.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        n = len(data)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        print(f"   Train: {len(train_data)} examples")
        print(f"   Val:   {len(val_data)} examples")
        print(f"   Test:  {len(test_data)} examples")
        
        return train_data, val_data, test_data
    
    def save_processed_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                           test_data: pd.DataFrame):
        """
        Save processed datasets to CSV files.
        """
        print("💾 Saving processed data...")
        
        train_path = self.processed_data_path / "train.csv"
        val_path = self.processed_data_path / "val.csv"
        test_path = self.processed_data_path / "test.csv"
        
        train_data.to_csv(train_path, index=False)
        val_data.to_csv(val_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        print(f"   ✅ Saved to {self.processed_data_path}/")
        print(f"      - train.csv ({len(train_data)} examples)")
        print(f"      - val.csv ({len(val_data)} examples)")
        print(f"      - test.csv ({len(test_data)} examples)")
    
    def compile_full_pipeline(self):
        """
        Run the complete data compilation pipeline.
        """
        print("=" * 60)
        print("🚀 STARTING DATA COMPILATION PIPELINE")
        print("=" * 60)
        
        # Step 1: Download/check datasets
        scam_files, legit_files = self.download_datasets()
        
        if not scam_files or not legit_files:
            print("\n❌ Cannot proceed without datasets.")
            print("Please add scam_*.csv and legit_*.csv files to data/raw/")
            return False
        
        # Step 2: Load data
        all_data = []
        
        for scam_file in scam_files:
            scam_data = self.load_scam_data(scam_file)
            all_data.append(scam_data)
        
        for legit_file in legit_files:
            legit_data = self.load_legitimate_data(legit_file)
            all_data.append(legit_data)
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"\n📊 Combined dataset: {len(combined_data)} total examples")
        
        # Step 3: Clean and format
        clean_data = self.clean_and_format(combined_data)
        
        # Step 4: Balance classes
        balanced_data = self.balance_classes(clean_data)
        
        # Step 5: Split data
        train_data, val_data, test_data = self.split_train_val_test(balanced_data)
        
        # Step 6: Save processed data
        self.save_processed_data(train_data, val_data, test_data)
        
        print("\n" + "=" * 60)
        print("✅ DATA COMPILATION COMPLETE")
        print("=" * 60)
        
        return True
    
    def get_dataset_stats(self):
        """
        Display statistics about the processed datasets.
        """
        print("\n📊 DATASET STATISTICS")
        print("-" * 60)
        
        for split in ['train', 'val', 'test']:
            file_path = self.processed_data_path / f"{split}.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                scam_count = len(df[df['label'] == 'scam'])
                legit_count = len(df[df['label'] == 'legit'])
                avg_length = df['text'].str.len().mean()
                
                print(f"\n{split.upper()} SET:")
                print(f"  Total examples: {len(df)}")
                print(f"  Scam: {scam_count} ({scam_count/len(df)*100:.1f}%)")
                print(f"  Legit: {legit_count} ({legit_count/len(df)*100:.1f}%)")
                print(f"  Avg text length: {avg_length:.0f} characters")
            else:
                print(f"\n{split.upper()} SET: Not found")
        
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description='Data Compiler for Scam Detection')
    parser.add_argument('--test', action='store_true', help='Run test mode to display stats')
    args = parser.parse_args()
    
    compiler = DataCompiler()
    
    if args.test:
        # Test mode: just show stats
        compiler.get_dataset_stats()
    else:
        # Full pipeline
        success = compiler.compile_full_pipeline()
        
        if success:
            compiler.get_dataset_stats()


if __name__ == "__main__":
    main()
