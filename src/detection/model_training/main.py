"""
Main orchestrator for Scam Detection ML Pipeline
Runs the complete pipeline: data compilation → training → evaluation
"""

import argparse
from pathlib import Path
from data_compiler import DataCompiler
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator


def check_requirements():
    """Check if required directories and files exist."""
    print("🔍 Checking requirements...")
    
    raw_data_path = Path("data/raw")
    if not raw_data_path.exists():
        print("❌ data/raw/ directory not found")
        return False
    
    scam_files = list(raw_data_path.glob("*scam*.csv"))
    legit_files = list(raw_data_path.glob("*legit*.csv"))
    
    if not scam_files:
        print("❌ No scam dataset found in data/raw/")
        return False
    
    if not legit_files:
        print("❌ No legitimate dataset found in data/raw/")
        return False
    
    print("✅ All requirements met")
    return True


def run_full_pipeline(epochs=3, batch_size=16):
    """Run the complete ML pipeline."""
    print("\n" + "=" * 70)
    print(" " * 15 + "SCAM DETECTION ML PIPELINE")
    print("=" * 70)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Pipeline cannot start. Please add datasets to data/raw/")
        return
    
    # Step 1: Data Compilation
    print("\n" + "🔹" * 35)
    print("STEP 1: DATA COMPILATION")
    print("🔹" * 35)
    
    compiler = DataCompiler()
    success = compiler.compile_full_pipeline()
    
    if not success:
        print("\n❌ Data compilation failed. Stopping pipeline.")
        return
    
    compiler.get_dataset_stats()
    
    # Step 2: Model Training
    print("\n" + "🔹" * 35)
    print("STEP 2: MODEL TRAINING")
    print("🔹" * 35)
    
    trainer = ModelTrainer()
    trainer.load_base_model()
    trainer.train_model(epochs=epochs, batch_size=batch_size)
    
    # Step 3: Model Evaluation
    print("\n" + "🔹" * 35)
    print("STEP 3: MODEL EVALUATION")
    print("🔹" * 35)
    
    evaluator = ModelEvaluator()
    metrics = evaluator.run_full_evaluation()
    
    # Final Summary
    print("\n" + "=" * 70)
    print(" " * 20 + "PIPELINE COMPLETE!")
    print("=" * 70)
    print("\n📊 Final Results:")
    print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"   Precision: {metrics['precision']*100:.2f}%")
    print(f"   Recall:    {metrics['recall']*100:.2f}%")
    print(f"   F1 Score:  {metrics['f1']*100:.2f}%")
    
    print("\n📁 Deliverables:")
    print("   ✅ Trained model: models/best_model/")
    print("   ✅ Confusion matrix: outputs/confusion_matrix.png")
    print("   ✅ Evaluation report: outputs/evaluation_report.json")
    
    print("\n🚀 Next Steps:")
    print("   → Share models/best_model/ with Samya for deployment")
    print("   → Send outputs/confusion_matrix.png to Nicole for dashboard")
    print("   → Review outputs/evaluation_report.json for detailed metrics")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Scam Detection ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --full-pipeline          # Run complete pipeline
  python main.py --compile-only           # Only compile data
  python main.py --train-only --epochs 5  # Only train model
  python main.py --evaluate-only          # Only evaluate model
        """
    )
    
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run complete pipeline (compile → train → evaluate)')
    parser.add_argument('--compile-only', action='store_true',
                       help='Only run data compilation')
    parser.add_argument('--train-only', action='store_true',
                       help='Only run model training')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only run model evaluation')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size (default: 16)')
    
    args = parser.parse_args()
    
    # If no specific flag is set, run full pipeline
    if not any([args.full_pipeline, args.compile_only, args.train_only, args.evaluate_only]):
        args.full_pipeline = True
    
    if args.full_pipeline:
        run_full_pipeline(epochs=args.epochs, batch_size=args.batch_size)
    
    elif args.compile_only:
        compiler = DataCompiler()
        compiler.compile_full_pipeline()
        compiler.get_dataset_stats()
    
    elif args.train_only:
        trainer = ModelTrainer()
        trainer.load_base_model()
        trainer.train_model(epochs=args.epochs, batch_size=args.batch_size)
    
    elif args.evaluate_only:
        evaluator = ModelEvaluator()
        evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
