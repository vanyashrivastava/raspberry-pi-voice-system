"""
Test Script for Scam Detection Model
Test your trained model on custom messages
"""

import pickle
import sys

def load_model():
    """Load the trained model and vectorizer."""
    try:
        with open('models/lightweight_model/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('models/lightweight_model/model.pkl', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    except FileNotFoundError:
        print("❌ Error: Model not found. Please train the model first.")
        print("   Run: python train_simple_model.py")
        sys.exit(1)

def predict_message(text, vectorizer, model):
    """Predict if a message is scam or legitimate."""
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    
    is_scam = prediction == 1
    confidence = probabilities[prediction] * 100
    
    return is_scam, confidence

def main():
    print("=" * 70)
    print(" " * 20 + "SCAM DETECTOR TEST")
    print("=" * 70)
    
    # Load model
    print("\n📦 Loading trained model...")
    vectorizer, model = load_model()
    print("   ✅ Model loaded successfully!")
    
    # Test mode selection
    print("\n" + "=" * 70)
    print("Choose test mode:")
    print("  1. Test predefined examples")
    print("  2. Test your own messages")
    print("  3. Interactive mode (type messages one by one)")
    print("=" * 70)
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Predefined examples
        test_messages = [
            ("Congratulations! You've won $1,000,000! Click here NOW!", True),
            ("Hi, can we reschedule our meeting to next Tuesday?", False),
            ("URGENT: Your account will be suspended. Verify immediately!", True),
            ("Thanks for sending the report. I'll review it tonight.", False),
            ("You have been selected for a FREE iPhone! Limited offer!", True),
            ("The project deadline is Friday. Let me know if you need help.", False),
            ("Your package is held at customs. Pay $50 fee to release it.", True),
            ("Looking forward to our call tomorrow at 3pm.", False),
            ("FINAL NOTICE: IRS tax payment required or face arrest.", True),
            ("Great job on the presentation today! Well done.", False),
        ]
        
        print("\n" + "=" * 70)
        print("🧪 TESTING PREDEFINED EXAMPLES")
        print("=" * 70)
        
        correct = 0
        for i, (msg, expected_scam) in enumerate(test_messages, 1):
            is_scam, confidence = predict_message(msg, vectorizer, model)
            
            result = "🚨 SCAM" if is_scam else "✅ LEGIT"
            expected = "scam" if expected_scam else "legit"
            match = "✓" if is_scam == expected_scam else "✗"
            
            if is_scam == expected_scam:
                correct += 1
            
            print(f"\n{i}. {msg[:60]}...")
            print(f"   Prediction: {result} ({confidence:.1f}% confident) {match}")
        
        accuracy = (correct / len(test_messages)) * 100
        print(f"\n{'=' * 70}")
        print(f"Accuracy: {correct}/{len(test_messages)} ({accuracy:.1f}%)")
        print("=" * 70)
    
    elif choice == "2":
        # Custom messages
        print("\n" + "=" * 70)
        print("Enter your messages (one per line, empty line to finish):")
        print("=" * 70)
        
        messages = []
        while True:
            msg = input("\nMessage: ").strip()
            if not msg:
                break
            messages.append(msg)
        
        if messages:
            print("\n" + "=" * 70)
            print("🧪 RESULTS")
            print("=" * 70)
            
            for i, msg in enumerate(messages, 1):
                is_scam, confidence = predict_message(msg, vectorizer, model)
                result = "🚨 SCAM" if is_scam else "✅ LEGIT"
                
                print(f"\n{i}. {msg[:60]}...")
                print(f"   → {result} ({confidence:.1f}% confident)")
        else:
            print("\nNo messages entered.")
    
    elif choice == "3":
        # Interactive mode
        print("\n" + "=" * 70)
        print("🔄 INTERACTIVE MODE (type 'quit' to exit)")
        print("=" * 70)
        
        while True:
            msg = input("\nTest message: ").strip()
            
            if msg.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not msg:
                continue
            
            is_scam, confidence = predict_message(msg, vectorizer, model)
            result = "🚨 SCAM" if is_scam else "✅ LEGIT"
            
            print(f"   → {result} ({confidence:.1f}% confident)")
    
    else:
        print("\n❌ Invalid choice. Please run again and select 1, 2, or 3.")

if __name__ == "__main__":
    main()
