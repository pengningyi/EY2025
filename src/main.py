import argparse
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from utils import load_data, scale_data, split_data
from mlp import MLPTrainer
from wideDeep import WideAndDeepTrainer
from tabnet import TabNetTrainer
from xgb_trainer import XGBoostTrainer

def main(args):
    os.makedirs(args.results_path, exist_ok=True)

    print("Loading data...")
    X, y = load_data(args.data_path)
    X = scale_data(X)

    if args.model_type == "MLP":
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, args.test_size, args.val_size, args.seed, data_type="tensor")
        trainer = MLPTrainer(input_dim=X_train.shape[1],
                             hidden_units=args.hidden_units,
                             dropout_rate=args.dropout_rate,
                             learning_rate=args.lr)
    elif args.model_type == "WideAndDeep":
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, args.test_size, args.val_size, args.seed, data_type="tensor")
        trainer = WideAndDeepTrainer(input_dim=X_train.shape[1],
                                     wide_units=args.wide_units,
                                     deep_units=args.deep_units,
                                     learning_rate=args.lr)
    elif args.model_type == "TabNet":
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, args.test_size, args.val_size, args.seed, data_type="tensor")
        trainer = TabNetTrainer()
    elif args.model_type == "XGBoost":
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, args.test_size, args.val_size, args.seed, data_type="dataframe")
        trainer = XGBoostTrainer()
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    print("Training the model...")
    trainer.fit(X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=args.batch_size)

    print("Evaluating the model...")
    test_rmse = trainer.evaluate(X_test, y_test)
    print(f"Test RMSE: {test_rmse:.4f}")

    results = {
        "model_type": args.model_type,
        "test_rmse": test_rmse,
        "parameters": vars(args)
    }
    results_path = os.path.join(args.results_path, f"{args.model_type}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models.")

    parser.add_argument("--data_path", type=str, default= '../data/training_data.csv', help="Path to the data file.")
    parser.add_argument("--results_path", type=str, default="../results", help="Directory to save results.")

    parser.add_argument("--model_type", type=str, required=True, choices=["MLP", "WideAndDeep", "TabNet", "XGBoost"], help="Type of model to train.")

    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to use as the test set.")
    parser.add_argument("--val_size", type=float, default=0.2, help="Proportion of the training set to use as the validation set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--hidden_units", type=int, nargs="*", default=[128, 64], help="Hidden units for MLP.")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for MLP.")
    parser.add_argument("--wide_units", type=int, default=64, help="Wide units for WideAndDeep.")
    parser.add_argument("--deep_units", type=int, nargs="*", default=[128, 64], help="Deep units for WideAndDeep.")
    parser.add_argument("--max_depth", type=int, default=6, help="Max depth for XGBoost.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators for XGBoost.")

    args = parser.parse_args()
    main(args)