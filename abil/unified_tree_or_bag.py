import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor, DMatrix
from sklearn.pipeline import Pipeline
from sklearn import base

# these are designed for internal use
from joblib import delayed, Parallel


def process_data_with_model(
    m, X_predict, X_train, y_train, cv=None, chunksize=None
):
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
    if isinstance(m, Pipeline):
        pipeline = m
        m = pipeline.named_steps["estimator"]
    else:
        pipeline = Pipeline([("preprocessor", FunctionTransformer()), ("estimator", m)])
    preprocessor = pipeline.named_steps["preprocessor"]

    X_train = preprocessor.transform(X_train)
    X_predict = preprocessor.transform(X_predict)

    # for internal, create models for each fold to mimic the
    # effect of leaving a fold out. Do this in parallel.

    if cv is not None:
        train_summary_stats = [
            _summarize_predictions(
                model,
                X_predict=X_train.iloc[test_idx],
                X_train=X_train.iloc[train_idx],
                y_train=y_train.iloc[train_idx],
                chunksize=chunksize,
            )
            for train_idx, test_idx in cv.split(X_train)
        ]
    # concat and rearrange to match input data order
        train_summary_stats  = pd.concat(
            train_summary_stats, axis=0, ignore_index=False
        ).loc[X_train.index]
    else:
        train_summary_stats = _summarize_predictions(
            model,
            X_train=X_train,
            y_train=y_train,
            X_predict=X_train,
            chunksize=chunksize
        )

    predict_summary_stats = _summarize_predictions(
        model, 
        X_predict=X_predict,
        chunksize=chunksize
    )
    
    return {"train_stats": train_summary_stats, "predict_stats": predict_summary_stats}


def _summarize_predictions(
    model, X_predict, X_train=None, y_train=None, chunksize=2e4
):
    # need to extract the ensemble predictions for each X
    # over all learners, then summarize those
    # and do that in parallel.
    if (X_train is not None) & (y_train is not None):
        model = base.clone(model).fit(X_train, y_train)
    elif not all([(X_train is None), (y_train is None)]):
        if not base.check_is_fitted(model):
            raise ValueError("model provided is not fit, and no data is provided to fit the model on. Fit the model on the data first.")

        raise ValueError(
            "Both X_train and y_train must be provided, or neither must be."
        )
        
    engine = Parallel()
    n_samples, n_features = X_predict.shape

    if chunksize is not None:
        n_chunks = int(np.ceil(n_samples / chunksize))
        chunks = np.array_split(X_predict, n_chunks)
    else:
        chunks = [X_predict]

    stats = []
    inverse_transform = getattr(
        model, "inverse_transform", FunctionTransformer().inverse_transform
    )
    for chunk in chunks:
        try:
            booster = model.get_booster()
            pred_jobs = (
                delayed(booster.predict)(DMatrix(chunk), iteration_range=(i, i + 1))
                for i in range(model.n_estimators)

            )

        except AttributeError:
            pred_jobs = (
                delayed(member.predict)(chunk)
                for member in getattr(model, "estimators_", [model])
            )
        chunk_preds = pd.DataFrame(
            inverse_transform(np.column_stack(engine(pred_jobs))),
            index = getattr(chunk, 'index', None)
            )
        chunk_stats = pd.DataFrame.from_dict(
            dict(
                mean = chunk_preds.mean(axis=1),
                sd = chunk_preds.std(axis=1),
                median = chunk_preds.std(axis=1),
                **dict(
                    zip(['ci95_LL', 'ci95_UL'], 
                        chunk_preds.quantile(q=(.025, .975), axis=1).values
                    )
                )
            )
        )
        stats.append(chunk_stats)
    output = pd.concat(
        stats, axis=0, ignore_index = False
    )
    return output
        
        
# Example Usage
if __name__ == "__main__":
    # Generate sample data
    from sklearn.datasets import make_regression
    from joblib import parallel_backend  # this is user-facing

    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    X_train = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    latitudes = np.random.uniform(-90, 90, size=X_train.shape[0])
    longitudes = np.random.uniform(-180, 180, size=X_train.shape[0])
    X_train.index = pd.MultiIndex.from_tuples(
        zip(latitudes, longitudes), names=["Latitude", "Longitude"]
    )
    y_train = pd.Series(y)

    X_predict, _ = make_regression(n_samples=20, n_features=10, noise=0.1)
    X_predict = pd.DataFrame(
        X_predict, columns=[f"feature_{i}" for i in range(X_predict.shape[1])]
    )
    latitudes_predict = np.random.uniform(-90, 90, size=X_predict.shape[0])
    longitudes_predict = np.random.uniform(-180, 180, size=X_predict.shape[0])
    X_predict.index = pd.MultiIndex.from_tuples(
        zip(latitudes_predict, longitudes_predict), names=["Latitude", "Longitude"]
    )

    cv_splits = 5
    # Define cross-validation strategy
    cv = KFold(n_splits=cv_splits)

    # Define model and method
    model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42).fit(X_train, y_train)
    # this sets the backend type and number of jobs to use in the internal
    # Parallel() call.
    with parallel_backend("loky", n_jobs=16):
        results = process_data_with_model(
            model, X_predict=X_predict, X_train=X_train, y_train=y_train, cv=cv
        )

    print("\n=== Training Summary Stats ===\n", results["train_stats"].head())
    print("\n=== Prediction Summary Stats ===\n", results["predict_stats"].head())
