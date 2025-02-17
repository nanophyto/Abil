import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor


def RFtree_stats(model_list, X, y, cv):
    """
    Compute out-of-sample predictions and summary statistics across all folds
    and trees in cross-validation using pre-trained RandomForest models.
    
    Parameters:
    - model_list: List of trained RandomForest models from each fold.
    - X: Feature matrix (pandas DataFrame).
    - y: Target vector (pandas Series).
    - cv: Cross-validation splitter (e.g., KFold object).
    
    Returns:
    - summary_stats: A DataFrame with mean, std, and confidence intervals
                     computed across all folds and trees for each sample.
    """
    all_fold_predictions = {}  # Store predictions across all folds for each sample

    for fold_idx, (_, test_idx) in enumerate(cv.split(X)):
        X_test = X.iloc[test_idx]
        rf_model = model_list[fold_idx]
        
        # Collect predictions for each tree in the current model
        fold_tree_predictions = []
        for tree in rf_model.estimators_:
            tree_preds = tree.predict(X_test)
            fold_tree_predictions.append(tree_preds)
        
        fold_tree_predictions = np.array(fold_tree_predictions).T  # shape: (n_samples, n_trees)
        
        # Store predictions for the test set
        for idx, test_sample_idx in enumerate(test_idx):
            if test_sample_idx not in all_fold_predictions:
                all_fold_predictions[test_sample_idx] = []
            all_fold_predictions[test_sample_idx].append(fold_tree_predictions[idx, :])
    
    # Combine predictions for each sample across all folds and trees
    combined_predictions = {}
    for sample_idx, preds_list in all_fold_predictions.items():
        # Combine all predictions for the sample (across folds and trees)
        combined_predictions[sample_idx] = np.concatenate(preds_list, axis=0)
    
    # Compute summary statistics for each sample
    summary_stats = []
    for sample_idx, preds in combined_predictions.items():
        mean_pred = np.mean(preds)
        std_pred = np.std(preds)
        lower_bound = np.quantile(preds, 0.025)
        upper_bound = np.quantile(preds, 0.975)
        
        summary_stats.append({
            'index': sample_idx,
            'mean': mean_pred,
            'sd': std_pred,
            'ci95_LL': lower_bound,
            'ci95_UL': upper_bound
        })
    
    # Convert summary statistics to a DataFrame
    summary_stats_df = pd.DataFrame(summary_stats).set_index('index').sort_index()
    
    return summary_stats_df


def predict_with_trees_cv(model_list, X_predict):
    """
    Make predictions for a new dataset (X_predict) using all trees and all folds
    with pre-trained RandomForest models. The predictions are aggregated to compute summary statistics.
    
    Parameters:
    - model_list: List of trained RandomForest models from each fold.
    - X_predict: Prediction feature matrix (pandas DataFrame).
    
    Returns:
    - summary_stats: A DataFrame with mean, std, and confidence intervals for each sample in X_predict.
    """
    all_fold_predictions = []  # Collect predictions across all folds and trees

    for rf_model in model_list:
        # Collect predictions for each tree in the current model
        fold_tree_predictions = []
        for tree in rf_model.estimators_:
            tree_preds = tree.predict(X_predict)
            fold_tree_predictions.append(tree_preds)
        
        # Store predictions for this fold across all trees
        all_fold_predictions.append(np.array(fold_tree_predictions).T)  # shape: (n_samples, n_trees)
    
    # Concatenate predictions across all folds
    all_fold_predictions = np.concatenate(all_fold_predictions, axis=1)  # shape: (n_samples, (n_folds * n_trees))
    
    # Calculate summary statistics
    mean_preds = np.mean(all_fold_predictions, axis=1)
    std_preds = np.std(all_fold_predictions, axis=1)
    lower_bound = np.quantile(all_fold_predictions, 0.025, axis=1)
    upper_bound = np.quantile(all_fold_predictions, 0.975, axis=1)
    
    summary_stats = pd.DataFrame({
        'mean': mean_preds,
        'sd': std_preds,
        'ci95_LL': lower_bound,
        'ci95_UL': upper_bound
    }, index=X_predict.index)  # Set index to match X_predict
    
    return summary_stats


if __name__ == "__main__":
    # Generate sample training data
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    X_train = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    # Generate random latitude and longitude and set as MultiIndex
    latitudes = np.random.uniform(-90, 90, size=X_train.shape[0])
    longitudes = np.random.uniform(-180, 180, size=X_train.shape[0])
    X_train.index = pd.MultiIndex.from_tuples(zip(latitudes, longitudes), names=['Latitude', 'Longitude'])

    y_train = pd.Series(y)
    n_splits = 5

    # Define the model and cross-validation strategy
    model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
    cv = KFold(n_splits=n_splits)

    # Train models once and store them
    model_list = []
    for train_idx, _ in cv.split(X_train):
        X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
        rf_model.fit(X_train_fold, y_train_fold)
        model_list.append(rf_model)

    # Step 1: Out-of-sample prediction statistics for data for which we have y
    oos_summary_stats = RFtree_stats(model_list, X_train, y_train, cv)

    # Generate new sample data for X_predict (this is the data for which we do not have y)
    X_predict, _ = make_regression(n_samples=20, n_features=10, noise=0.1)
    X_predict = pd.DataFrame(X_predict, columns=[f"feature_{i}" for i in range(X_predict.shape[1])])

    # Generate random latitude and longitude and set as MultiIndex for X_predict
    latitudes_predict = np.random.uniform(-90, 90, size=X_predict.shape[0])
    longitudes_predict = np.random.uniform(-180, 180, size=X_predict.shape[0])
    X_predict.index = pd.MultiIndex.from_tuples(zip(latitudes_predict, longitudes_predict), names=['Latitude', 'Longitude'])

    # Step 2: Make predictions for X_predict
    predict_summary_stats = predict_with_trees_cv(model_list, X_predict)

    # Check predictions
    print("\nOut-of-sample Predictions \n", oos_summary_stats.head())
    print("\nPredictions for New Data (X_predict):\n", predict_summary_stats.head())
