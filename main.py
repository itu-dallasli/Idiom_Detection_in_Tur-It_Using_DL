import os
import argparse
import pandas as pd
import torch
from dataset import get_dataloaders
from model import train_model, BertForIdiomDetection, predict_idioms
from transformers import BertTokenizer


def run_train(args):
    train_loader, val_loader, tokenizer = get_dataloaders(
        train_path='public_data/train.csv',
        val_path='public_data/eval.csv',
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    model = train_model(
        train_loader, val_loader, tokenizer,
        epochs=args.epochs, lr=args.lr
    )
    print('Training complete. Best model saved as best_idiom_model.pt')


def run_eval(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, tokenizer = get_dataloaders(
        train_path='public_data/train.csv',
        val_path='public_data/eval.csv',
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    model = BertForIdiomDetection()
    model.load_state_dict(torch.load('best_idiom_model.pt', map_location=device))
    model.to(device)
    from model import evaluate
    metrics = evaluate(model, val_loader, tokenizer, device)
    print('Evaluation complete.')
    print(metrics)


def run_predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForIdiomDetection()
    model.load_state_dict(torch.load('best_idiom_model.pt', map_location=device))
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Read test data
    test_df = pd.read_csv('starting_kit/eval_w_o_labels.csv')
    ids = test_df['id'].tolist()
    sentences = test_df['sentence'].tolist()

    results = []
    for idx, sentence in zip(ids, sentences):
        _, idiom_indices = predict_idioms(model, tokenizer, sentence, device)
        # If no idiom, output [-1] as in training
        if not idiom_indices:
            idiom_indices = [-1]
        results.append({'id': idx, 'indices': str(idiom_indices)})

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f'Predictions saved to {args.output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval', 'predict'], required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--output', type=str, default='predictions.csv')
    args = parser.parse_args()

    if args.mode == 'train':
        run_train(args)
    elif args.mode == 'eval':
        run_eval(args)
    elif args.mode == 'predict':
        run_predict(args)
    else:
        raise ValueError('Unknown mode')

if __name__ == '__main__':
    main() 