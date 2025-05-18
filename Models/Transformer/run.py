import argparse
import os
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Run MWE detection model training and evaluation')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict', 'both'],
                      help='Mode to run: train, predict, or both')
    parser.add_argument('--train_path', type=str, help='Path to training data CSV')
    parser.add_argument('--val_path', type=str, help='Path to validation data CSV')
    parser.add_argument('--test_path', type=str, help='Path to test data CSV')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save model, config, and predictions')
    parser.add_argument('--model_path', type=str, help='Path to trained model weights (for predict mode)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def run_command(command):
    print(f"\nExecuting: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error executing command:")
        print(result.stderr)
        sys.exit(1)
    print(result.stdout)

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode in ['train', 'both']:
        if not args.train_path or not args.val_path:
            print("Error: train_path and val_path are required for training")
            sys.exit(1)
            
        # Run training
        train_command = [
            'python', 'train_prime.py',
            '--train_path', args.train_path,
            '--val_path', args.val_path,
            '--output_dir', args.output_dir,
            '--seed', str(args.seed)
        ]
        run_command(train_command)
    
    if args.mode in ['predict', 'both']:
        if not args.test_path:
            print("Error: test_path is required for prediction")
            sys.exit(1)
            
        # Use model path from training if not specified
        model_path = args.model_path or os.path.join(args.output_dir, 'final_model.pt')
        config_path = os.path.join(args.output_dir, 'training_config.json')
        output_path = os.path.join(args.output_dir, 'predictions.csv')
        
        # Run prediction
        predict_command = [
            'python', 'predict_prime.py',
            '--input_path', args.test_path,
            '--model_path', model_path,
            '--output_path', output_path,
            '--config_path', config_path
        ]
        run_command(predict_command)

if __name__ == "__main__":
    main() 