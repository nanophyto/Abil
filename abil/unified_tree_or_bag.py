import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor, DMatrix


def train_model(model, X, y, train_idx):
    """
    Train the model for the given train indices.
    """
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    model.fit(X_train, y_train)
    return model


def collect_predictions(model, X_test, method):
    """
    Collect predictions for each "unit" (tree, bag, etc.) from the model.
    
    Parameters:
    - model: The trained model.
    - X_test: Test dataset (pandas DataFrame).
    - method: Specifies the type of model ("xgb", "rf", "bagging").
    
    Returns:
    - Predictions as a NumPy array with shape (n_samples, n_units).
    """
    if method == "xgb":
        booster = model.get_booster()
        X_test_dmatrix = DMatrix(X_test)
        predictions = []
        for i in range(model.n_estimators):
            preds = booster.predict(X_test_dmatrix, iteration_range=(i, i + 1))
            predictions.append(preds)
        return np.array(predictions).T  # (n_samples, n_trees)

    elif method == "rf":
        predictions = []
        for tree in model.estimators_:
            preds = tree.predict(X_test)
            predictions.append(preds)
        return np.array(predictions).T  # (n_samples, n_trees)

    elif method == "bagging":
        predictions = []
        for bag in model.estimators_:
            preds = bag.predict(X_test)
            predictions.append(preds)
        return np.array(predictions).T  # (n_samples, n_bags)

    else:
        raise ValueError("Unsupported method. Choose from 'xgb', 'rf', or 'bagging'.")


def compute_summary_stats(predictions, indices):
    """
    Compute summary statistics (mean, std, CI) for predictions.
    """
    mean_preds = np.mean(predictions, axis=1)
    std_preds = np.std(predictions, axis=1)
    lower_bound = np.quantile(predictions, 0.025, axis=1)
    upper_bound = np.quantile(predictions, 0.975, axis=1)

    return pd.DataFrame({
        'index': indices,
        'mean': mean_preds,
        'sd': std_preds,
        'ci95_LL': lower_bound,
        'ci95_UL': upper_bound
    }).set_index('index')


def cross_val_predictions(model, X, y, cv, method):
    """
    Perform cross-validation and compute predictions with summary statistics.
    """
    all_predictions = {}
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        trained_model = train_model(model, X, y, train_idx)
        X_test = X.iloc[test_idx]
        fold_preds = collect_predictions(trained_model, X_test, method)
        
        for idx, sample_idx in enumerate(test_idx):
            if sample_idx not in all_predictions:
                all_predictions[sample_idx] = []
            all_predictions[sample_idx].append(fold_preds[idx, :])
    
    combined_predictions = {
        idx: np.concatenate(pred_list, axis=0)
        for idx, pred_list in all_predictions.items()
    }
    return compute_summary_stats(
        np.array(list(combined_predictions.values())),
        np.array(list(combined_predictions.keys()))
    )


def predict_new_data(model, model_list, X_predict, method):
    """
    Make predictions for new data using the provided models and compute summary statistics.
    """
    all_predictions = []
    for trained_model in model_list:
        fold_preds = collect_predictions(trained_model, X_predict, method)
        all_predictions.append(fold_preds)
    
    all_predictions = np.concatenate(all_predictions, axis=1)  # Combine folds
    return compute_summary_stats(all_predictions, X_predict.index)


if __name__ == "__main__":
    # Generate sample training data
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    X_train = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    latitudes = np.random.uniform(-90, 90, size=X_train.shape[0])
    longitudes = np.random.uniform(-180, 180, size=X_train.shape[0])
    X_train.index = pd.MultiIndex.from_tuples(zip(latitudes, longitudes), names=['Latitude', 'Longitude'])
    y_train = pd.Series(y)

    # Generate new sample data for X_predict
    X_predict, _ = make_regression(n_samples=20, n_features=10, noise=0.1)
    X_predict = pd.DataFrame(X_predict, columns=[f"feature_{i}" for i in range(X_predict.shape[1])])
    latitudes_predict = np.random.uniform(-90, 90, size=X_predict.shape[0])
    longitudes_predict = np.random.uniform(-180, 180, size=X_predict.shape[0])
    X_predict.index = pd.MultiIndex.from_tuples(zip(latitudes_predict, longitudes_predict), names=['Latitude', 'Longitude'])

    # Define CV strategy
    n_splits = 5
    cv = KFold(n_splits=n_splits)

    # Define and evaluate models
    models = {
        "xgb": XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
        "rf": RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42),
        "bagging": BaggingRegressor(estimator=KNeighborsRegressor(), n_estimators=50, random_state=42)
    }

    for method, model in models.items():
        print(f"\n=== {method.upper()} ===")
        model_list = [train_model(model, X_train, y_train, train_idx) for train_idx, _ in cv.split(X_train)]
        oos_summary_stats = cross_val_predictions(model, X_train, y_train, cv, method)
        print("\nOut-of-sample Predictions:\n", oos_summary_stats.head())
        predict_summary_stats = predict_new_data(model, model_list, X_predict, method)
        print("\nPredictions for New Data:\n", predict_summary_stats.head())
