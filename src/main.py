import argparse
import os
import json
from utils import load_data, scale_data, split_data, generate_submission
from mlp import MLPTrainer
from wideDeep import WideAndDeepTrainer
from tabnet import TabNetTrainer
from xgb_trainer import XGBoostTrainer
from modelBlender import ModelBlender
from cnn import CNN1DTrainer
from data import get_sentinel_data
import numpy as np

def convert_float32_to_float(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_float32_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_float32_to_float(v) for v in obj]
    else:
        return obj

def main(args):
    os.makedirs(args.results_path, exist_ok=True)
    
    print("Loading data...")
    X, y = load_data(get_sentinel_data(args.data_path))
    X = scale_data(X)

    if args.model_type == "MLP":
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, args.test_size, args.val_size, args.seed, data_type="tensor")
        trainer = MLPTrainer(input_dim=X_train.shape[1],
                             hidden_units=args.hidden_units,
                             dropout_rate=args.dropout_rate,
                             learning_rate=args.lr)
        results_path = os.path.join(args.results_path, f"{args.model_type}_{args.epochs}_{args.batch_size}_{args.lr}_{args.seed}_results.json")
    elif args.model_type == "WideAndDeep":
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, args.test_size, args.val_size, args.seed, data_type="tensor")
        trainer = WideAndDeepTrainer(input_dim=X_train.shape[1],
                                     wide_units=args.wide_units,
                                     deep_units=args.deep_units,
                                     learning_rate=args.lr)
        results_path = os.path.join(args.results_path, f"{args.model_type}_{args.epochs}_{args.batch_size}_{args.lr}_{args.seed}_results.json")
    elif args.model_type == "TabNet":
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, args.test_size, args.val_size, args.seed, data_type="tensor")
        trainer = TabNetTrainer()
        results_path = os.path.join(args.results_path, f"{args.model_type}_{args.epochs}_{args.batch_size}_{args.lr}_{args.seed}_results.json")
    elif args.model_type == "XGBoost":
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, args.test_size, args.val_size, args.seed, data_type="dataframe")
        trainer = XGBoostTrainer(args.lr, args.n_estimators, args.max_depth,
                                 args.subsample, args.colsample_bytree,
                                 args.reg_alpha, args.reg_lambda, args.seed)
        results_path = os.path.join(args.results_path, f"{args.model_type}_{args.n_estimators}_{args.max_depth}_{args.lr}_{args.subsample}_{args.colsample_bytree}_{args.reg_alpha}_{args.reg_lambda}_{args.seed}_results.json")
    elif args.model_type == "CNN1D":
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, args.test_size, args.val_size, args.seed, data_type="tensor")
        trainer = CNN1DTrainer(input_dim=X_train.shape[1],
                               hidden_units=args.hidden_units,
                               dropout_rate=args.dropout_rate,
                               learning_rate=args.lr)
        results_path = os.path.join(args.results_path, f"{args.model_type}_{args.hidden_units}_{args.epochs}_{args.batch_size}_{args.lr}_{args.seed}_results.json")
    elif args.model_type == "ModelBlender":
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, args.test_size, args.val_size, args.seed, data_type="dataframe")
        trainer = ModelBlender()
        results_path = os.path.join(args.results_path, f"{args.model_type}_{args.epochs}_{args.batch_size}_{args.lr}_{args.seed}_results.json")
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    print("Training the model...")
    trainer.fit(X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=args.batch_size)

    print("Evaluating the model...")
    test_rmse = trainer.evaluate(X_test, y_test)
    print(f"Test RMSE: {test_rmse:.8f}")

    results = {
        "model_type": args.model_type,
        "test_rmse": test_rmse,
        "parameters": vars(args)
    }
    
    results = convert_float32_to_float(results)
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")
    
    print("Generating submission")
    generate_submission(trainer, args.submission_data_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models.")

    parser.add_argument("--data_path", type=str, default= 'C:\\Users\\16539\\Documents\\Aalto\\25spring\\EY2025\\data\\Training_data_uhi_index_2025-02-18.csv', help="Path to the data file.")
    parser.add_argument("--submission_data_path", type=str, default= '../data/Submission_template.csv', help="Path to the submission file.")
    parser.add_argument("--results_path", type=str, default="../results", help="Directory to save results.")

    parser.add_argument("--model_type", type=str, required=True, choices=["MLP", "WideAndDeep", "TabNet", "XGBoost","ModelBlender", "CNN1D"], help="Type of model to train.")

    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of the dataset to use as the test set.")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of the training set to use as the validation set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
 
    parser.add_argument("--hidden_units", type=int, nargs="*", default=[64, 128], help="Hidden units for MLP.")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate for MLP.")
    
    parser.add_argument("--wide_units", type=int, default=64, help="Wide units for WideAndDeep.")
    parser.add_argument("--deep_units", type=int, nargs="*", default=[64, 64], help="Deep units for WideAndDeep.")
    
    parser.add_argument("--max_depth", type=int, default=10, help="Max depth for XGBoost.")
    parser.add_argument("--n_estimators", type=int, default=500, help="Number of estimators for XGBoost.")
    parser.add_argument("--subsample", type=float, default=1, help="")
    parser.add_argument("--colsample_bytree", type=float, default=1, help="")
    parser.add_argument("--reg_alpha", type=float, default=0, help="")
    parser.add_argument("--reg_lambda", type=float, default=1, help="")
    
    
    args = parser.parse_args()
    main(args)