import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from xgboost import XGBRegressor, DMatrix

def XGBtree_stats(model, X, y, cv):
    """
    In this script the Booster object is accessed form XGBoost using model.booster.
    This booster object has an argument `iteration_range` which allows us to specify which
    trees are used for prediction, and is then used to loop through each tree.

    To get out-of-sample predictions for all our X_train data (data for which we have y's):
    we perform training on the first CV split, collect predictions from each tree,
    and compute quantiles for each sample in the out-of-sample set.
    Then the model trained on the next fold to predict on the first fold's test data.
    """
    # Get the first two splits from KFold
    train_idx_1, test_idx_1 = next(cv.split(X))
    train_idx_2, test_idx_2 = next(cv.split(X))
    
    # First fold: train and get out-of-sample predictions
    X_train_fold_1, X_test_fold_1 = X.iloc[train_idx_1], X.iloc[test_idx_1]
    y_train_fold_1 = y.iloc[train_idx_1]
    model.fit(X_train_fold_1, y_train_fold_1)

    booster_1 = model.get_booster()
    fold_tree_predictions_1 = []  # Stores each tree's predictions for the test set of the first fold

    X_test_dmatrix_1 = DMatrix(X_test_fold_1)

    for i in range(model.n_estimators):
        # Predict with the i-th tree only for X_test_fold_1
        tree_preds_1 = booster_1.predict(X_test_dmatrix_1, iteration_range=(i, i + 1))
        fold_tree_predictions_1.append(tree_preds_1)

    fold_tree_predictions_1 = np.array(fold_tree_predictions_1).T  # shape: (n_samples, n_trees)

    # Calculate summary statistics for the first fold's out-of-sample predictions
    mean_preds_1 = np.mean(fold_tree_predictions_1, axis=1)
    std_preds_1 = np.std(fold_tree_predictions_1, axis=1)
    lower_bound_1 = np.quantile(fold_tree_predictions_1, 0.025, axis=1)
    upper_bound_1 = np.quantile(fold_tree_predictions_1, 0.975, axis=1)

    summary_stats_1 = pd.DataFrame({
        'mean': mean_preds_1,
        'sd': std_preds_1,
        'ci95_LL': lower_bound_1,
        'ci95_UL': upper_bound_1
    }, index=X_test_fold_1.index)

    # Second fold: train on the second fold, then predict on X_test_fold_1 from the first fold
    X_train_fold_2, X_test_fold_2 = X.iloc[train_idx_2], X.iloc[test_idx_2]
    y_train_fold_2 = y.iloc[train_idx_2]
    model.fit(X_train_fold_2, y_train_fold_2)

    booster_2 = model.get_booster()
    fold_tree_predictions_2 = []  # Stores each tree's predictions for X_test_fold_1 using the second fold's model

    for i in range(model.n_estimators):
        # Predict with the i-th tree only for X_test_fold_1 using the second model
        tree_preds_2 = booster_2.predict(X_test_dmatrix_1, iteration_range=(i, i + 1))
        fold_tree_predictions_2.append(tree_preds_2)

    fold_tree_predictions_2 = np.array(fold_tree_predictions_2).T  # shape: (n_samples, n_trees)

    # Calculate summary statistics for predictions of X_test_fold_1 from the second fold's model
    mean_preds_2 = np.mean(fold_tree_predictions_2, axis=1)
    std_preds_2 = np.std(fold_tree_predictions_2, axis=1)
    lower_bound_2 = np.quantile(fold_tree_predictions_2, 0.025, axis=1)
    upper_bound_2 = np.quantile(fold_tree_predictions_2, 0.975, axis=1)

    summary_stats_2 = pd.DataFrame({
        'mean': mean_preds_2,
        'sd': std_preds_2,
        'ci95_LL': lower_bound_2,
        'ci95_UL': upper_bound_2
    }, index=X_test_fold_1.index)

    print("\nOut-of-sample Predictions from First Fold:\n", summary_stats_1.head())
    print("\nPredictions for First Fold Test Data using Second Fold Model:\n", summary_stats_2.head())
    
    return summary_stats_1, summary_stats_2

def predict_with_trees(model, X_predict):
    """
    Make predictions for a new dataset (X_predict for which we do not have y's) 
    using each individual tree in the model.
    """
    booster = model.get_booster()
    fold_tree_predictions = []  # Stores each tree's predictions for X_predict

    # Convert X_new to DMatrix
    X_predict_dmatrix = DMatrix(X_predict)

    for i in range(model.n_estimators):
        # Predict with the i-th tree only for X_predict
        tree_preds = booster.predict(X_predict_dmatrix, iteration_range=(i, i + 1))
        fold_tree_predictions.append(tree_preds)

    fold_tree_predictions = np.array(fold_tree_predictions).T  # shape: (n_samples, n_trees)

    # Calculate summary statistics
    mean_preds = np.mean(fold_tree_predictions, axis=1)
    std_preds = np.std(fold_tree_predictions, axis=1)
    lower_bound = np.quantile(fold_tree_predictions, 0.025, axis=1)
    upper_bound = np.quantile(fold_tree_predictions, 0.975, axis=1)

    summary_stats = pd.DataFrame({
        'mean': mean_preds,
        'sd': std_preds,
        'ci95_LL': lower_bound,
        'ci95_UL': upper_bound
    }, index=X_predict.index)  # Set index to match X_new

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
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    cv = KFold(n_splits=n_splits)

    # Step 1: Out-of-sample prediction statistics for data for which we have y
    oos_summary_stats_1, next_fold_summary_stats = XGBtree_stats(model, X_train, y_train, cv)

    # Generate new sample data for X_predict (this is the data for which we do not have y)
    X_predict, _ = make_regression(n_samples=20, n_features=10, noise=0.1)
    X_predict = pd.DataFrame(X_predict, columns=[f"feature_{i}" for i in range(X_predict.shape[1])])

    # Generate random latitude and longitude and set as MultiIndex for X_predict
    latitudes_predict = np.random.uniform(-90, 90, size=X_predict.shape[0])
    longitudes_predict = np.random.uniform(-180, 180, size=X_predict.shape[0])
    X_predict.index = pd.MultiIndex.from_tuples(zip(latitudes_predict, longitudes_predict), names=['Latitude', 'Longitude'])

    # Step 2: Make predictions for X_predict
    predict_summary_stats = predict_with_trees(model, X_predict)

    # Check predictions
    print("\nOut-of-sample Predictions for First Fold Test Data:\n", oos_summary_stats_1.head())
    print("\nPredictions for First Fold Test Data using Second Fold Model:\n", next_fold_summary_stats.head())
    print("\nPredictions for New Data (X_predict):\n", predict_summary_stats.head())
