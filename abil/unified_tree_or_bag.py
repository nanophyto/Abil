import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor, DMatrix


def process_data_with_model(X_train, y_train, X_predict, m, cv, cv_splits=5, zir=False, method="rf"):
    """
    Train the model using cross-validation, compute predictions on X_train with summary stats,
    and predict on X_predict with summary stats.

    Parameters:
    ----------
    X_train : DataFrame
        Training feature set with MultiIndex for coordinates.

    y_train : Series
        Target values corresponding to X_train.

    X_predict : DataFrame
        Feature set to predict on, with MultiIndex for coordinates.

    m : sklearn pipeline

    cv_splits : int, default=5
        Number of cross-validation splits.

    method : str, default="rf"
        Method type for handling different model-specific behaviors:
        "rf" for RandomForestRegressor,
        "bagging" for BaggingRegressor,
        "xgb" for XGBRegressor.

    Returns:
    -------
    dict
        Dictionary containing summary statistics for both training and prediction datasets.
        Keys: "train_stats", "predict_stats".
    """
    pipeline = m.regressor_

    preprocessor = pipeline.named_steps['preprocessor']
    X_train = preprocessor.transform(X_train)
    X_predict = preprocessor.transform(X_predict)

    if zir==False:
        estimator = pipeline.named_steps['estimator']
    else:
        clf_estimator = m.classifier_.named_steps['estimator']
        reg_estimator = m.regressor_.named_steps['estimator']

    def train_model(model, X, y, train_idx):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        if isinstance(model, BaggingRegressor):
            model = BaggingRegressor(
                estimator=KNeighborsRegressor(),
                n_estimators=model.n_estimators,
                random_state=model.random_state
            )
        model.fit(X_train_fold, y_train_fold)
        return model

    def collect_predictions(model, X_test, method):
        if method == "xgb":
            booster = model.get_booster()
            X_test_dmatrix = DMatrix(X_test)
            predictions = [
                booster.predict(X_test_dmatrix, iteration_range=(i, i + 1))
                for i in range(model.n_estimators)
            ]
            predictions = m.inverse_transform(predictions)  
            return np.array(predictions).T  # (n_samples, n_trees)

        elif method == "rf":
            predictions = [tree.predict(X_test) for tree in model.estimators_]
            predictions = m.inverse_transform(predictions)  
            return np.array(predictions).T  # (n_samples, n_trees)

        elif method == "bagging":
            predictions = [bag.predict(X_test) for bag in model.estimators_]
            predictions = m.inverse_transform(predictions)  
            return np.array(predictions).T  # (n_samples, n_bags)

        else:
            raise ValueError("Unsupported method. Choose from 'xgb', 'rf', or 'bagging'.")

    def compute_summary_stats(predictions, indices):
        mean_preds = np.mean(predictions, axis=1)
        std_preds = np.std(predictions, axis=1)
        lower_bound = np.quantile(predictions, 0.025, axis=1)
        upper_bound = np.quantile(predictions, 0.975, axis=1)
        stats_df = pd.DataFrame({
            'mean': mean_preds,
            'sd': std_preds,
            'ci95_LL': lower_bound,
            'ci95_UL': upper_bound
        }, index=indices)
        return stats_df

    def cross_val_predictions(model, X, y, cv, method):
        all_predictions = {}
        for train_idx, test_idx in cv.split(X, y):
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
        indices = [X.index[i] for i in combined_predictions.keys()]
        return compute_summary_stats(
            np.array(list(combined_predictions.values())),
            indices
        )

    def predict_new_data(model, model_list, X_predict, method):
        all_predictions = []
        for trained_model in model_list:
            fold_preds = collect_predictions(trained_model, X_predict, method)
            all_predictions.append(fold_preds)
        all_predictions = np.concatenate(all_predictions, axis=1)  # Combine folds
        return compute_summary_stats(all_predictions, X_predict.index)

    # Cross-validate on training data
    oos_summary_stats = cross_val_predictions(model, X_train, y_train, cv, method)

    # Train models for predictions on new data
    model_list = [train_model(model, X_train, y_train, train_idx) for train_idx, _ in cv.split(X_train)]
    predict_summary_stats = predict_new_data(model, model_list, X_predict, method)

    return {
        "train_stats": oos_summary_stats,
        "predict_stats": predict_summary_stats
    }


# Example Usage
if __name__ == "__main__":
    # Generate sample data
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    X_train = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    latitudes = np.random.uniform(-90, 90, size=X_train.shape[0])
    longitudes = np.random.uniform(-180, 180, size=X_train.shape[0])
    X_train.index = pd.MultiIndex.from_tuples(zip(latitudes, longitudes), names=['Latitude', 'Longitude'])
    y_train = pd.Series(y)

    X_predict, _ = make_regression(n_samples=20, n_features=10, noise=0.1)
    X_predict = pd.DataFrame(X_predict, columns=[f"feature_{i}" for i in range(X_predict.shape[1])])
    latitudes_predict = np.random.uniform(-90, 90, size=X_predict.shape[0])
    longitudes_predict = np.random.uniform(-180, 180, size=X_predict.shape[0])
    X_predict.index = pd.MultiIndex.from_tuples(zip(latitudes_predict, longitudes_predict), names=['Latitude', 'Longitude'])

    cv_splits = 5
    # Define cross-validation strategy
    cv = KFold(n_splits=cv_splits)

    # Define model and method
    model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
    results = process_data_with_model(X_train, y_train, X_predict, model, cv, cv_splits=cv_splits, method="rf")

    print("\n=== Training Summary Stats ===\n", results["train_stats"].head())
    print("\n=== Prediction Summary Stats ===\n", results["predict_stats"].head())
