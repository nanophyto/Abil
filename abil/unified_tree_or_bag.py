import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_absolute_error, balanced_accuracy_score

from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    BaggingRegressor,
    BaggingClassifier,
    VotingRegressor,
    VotingClassifier,
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import base
from joblib import delayed, Parallel

from sklearn.base import is_regressor

from .zir import ZeroInflatedRegressor
from . import utils as u
from sklearn.base import _is_fitted
from xgboost import XGBClassifier, XGBRegressor, DMatrix


def estimate_prediction_quantiles(
    model, X_predict, X_train, y_train, cv=None, chunksize=20_000, threshold=0.5
):
    """
    Train the model using cross-validation, compute predictions on X_train with summary stats,
    and predict on X_predict with summary stats.

    Parameters:
    -----------
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
    --------
    dict
        Dictionary containing summary statistics for both training and prediction datasets.
        Keys: "train_stats", "predict_stats".
    """

    #########################################
    # IF ZIR ESTIMATE QUANTILES RECURSIVELY #
    #########################################

    if isinstance(model, ZeroInflatedRegressor):
        original_y_train = y_train.copy()

        classifier_stats = estimate_prediction_quantiles(
            model.classifier_,
            X_predict = X_predict,
            X_train = X_train,
            y_train = (y_train > 0).copy(),
            cv=cv,
            chunksize=chunksize,
            threshold=threshold
        )
        regressor_stats = estimate_prediction_quantiles(
            model.regressor_, 
            X_predict=X_predict, 
            X_train=X_train, 
            y_train=original_y_train, 
            cv=cv, 
            chunksize=chunksize, 
            threshold=threshold
        )
        return {
            **{f"classifier_{k}": v for k, v in classifier_stats.items()},
            **{f"regressor_{k}": v for k, v in regressor_stats.items()},
        }

    #######################################
    # EXTRACT BASE MODEL AND TRANSFORMERS #
    #######################################

    if isinstance(model, Pipeline):
        print("model is classifier")
        pipeline = model
        model = pipeline.named_steps["estimator"]
        y_inverse_transformer = u.do_nothing
        y_transformer = u.do_nothing
    elif isinstance(model, TransformedTargetRegressor):
        pipeline = model.regressor
        y_transformer = getattr(
            model, "func", FunctionTransformer().func
        )
        print("y transformer is :", y_transformer)
        print("if log should be ~6.9: ", y_transformer(np.array([[1000]])))

        y_inverse_transformer = getattr(
            model, "inverse_func", FunctionTransformer().inverse_func
        )       
        print("y inverse transformer is :", y_inverse_transformer)
        print("if exp should be ~1000: ", y_inverse_transformer(np.array([[6.9]])))

        model = model.regressor.named_steps["estimator"]
    else:
        print("model is Voting regresor")
        pipeline = Pipeline(
            [("preprocessor", FunctionTransformer()), ("estimator", model)]
        )
        y_inverse_transformer = u.do_nothing
        y_transformer = u.do_nothing

    preprocessor = pipeline.named_steps["preprocessor"]
    preprocessor.fit(X_train)

    ##################################################
    # CHECK IF BASE MODEL IS REGRESSOR OR CLASSIFIER #
    ##################################################

    if is_regressor(model):
        proba = False
        print("model is regressor")
    else:
        proba = True
        print("model is classifier")
    

    ############################################
    # TRANSFORM Y_TRAIN, X_TRAIN AND X_PREDICT #
    ############################################

    if X_train is not None:
        X_train = pd.DataFrame(
            preprocessor.transform(X_train),
            index = getattr(X_train, "index", np.arange(X_train.shape[0])),
            columns=X_train.columns  
        )
    else:
        raise ValueError("X_train cannot be None")
    if X_predict is not None:
        X_predict = pd.DataFrame(
            preprocessor.transform(X_predict),
            index = getattr(X_predict, "index", np.arange(X_predict.shape[0])),
            columns=X_predict.columns  
        )
    else:
        raise ValueError("X_predict cannot be None")

    if y_train is not None:
        if not proba:
            print("max of y before transform is: ", np.max(y_train))

            y_train = pd.Series(                    
                y_transformer(y_train),
                index = getattr(y_train, "index", np.arange(y_train.shape[0]))
            )
    
            print("max of y after transform is: ", np.max(y_train))
          
        else: 
            print("model is classifier")
            y_train = pd.Series(
                y_train,
                index = getattr(y_train, "index", np.arange(y_train.shape[0]))
            )
    else:
        raise ValueError("y_train cannot be None")
    
    ##########################################
    # REFIT MODELS WITH TRANSFORMED Y AND Xs #
    ##########################################

    if not _is_fitted(model):
        model.fit(X_train, y_train)

        print("model is refitted with transformed data")

    ###############################################
    # MAKE QUANTILE PREDICTIONS FOR TRAINING DATA #
    ###############################################

    if cv is not None:
        print("CV is used in this model")
        train_summary_stats = [
            _summarize_predictions(
                model,
                y_inverse_transformer,
                X_predict=X_train.iloc[test_idx],
                X_train=X_train.iloc[train_idx],
                y_train=y_train.iloc[train_idx],
                chunksize=chunksize,
                threshold=threshold,
                proba=proba
            )
            for train_idx, test_idx in cv.split(X_train, y_train)
        ]
        # concat and rearrange to match input data order
        train_summary_stats = pd.concat(
            train_summary_stats, axis=0, ignore_index=False
        ).loc[X_train.index]
    else:
        print("CV is not used in this model")
        train_summary_stats = _summarize_predictions(
            model,
            y_inverse_transformer,
            X_train=X_train,
            X_predict=X_train,
            y_train=y_train,
            chunksize=chunksize,
            threshold=threshold,
            proba=proba
        )

    #################################################
    # MAKE QUANTILE PREDICTIONS FOR PREDICTION DATA #
    #################################################

    predict_summary_stats = _summarize_predictions(
        model, 
        y_inverse_transformer, 
        X_predict=X_predict, 
        X_train=X_train,
        y_train=y_train,
        chunksize=chunksize, 
        threshold=threshold,
        proba=proba
    )

    return {"train_stats": train_summary_stats, "predict_stats": predict_summary_stats}

#######################################
# DEFINE QUANTILE PREDICTION PIPELINE #
#######################################

def _summarize_predictions(model, y_inverse_transformer, X_predict, X_train=None, y_train=None, chunksize=2e4, threshold=0.5, proba=None):
    if proba is None:
        raise ValueError("proba not defined!")
    n_samples, n_features = X_predict.shape

    ######################################
    # DEFINE CHUNKS (FOR X_PREDICT ONLY) #
    ######################################

    if chunksize is not None:
        n_chunks = int(np.ceil(n_samples / chunksize))
        chunks = np.array_split(X_predict, n_chunks)
        # Convert each chunk to pandas DataFrame
        chunks = [pd.DataFrame(chunk, columns=X_predict.columns) for chunk in chunks]
    else:
        chunks = [X_predict]

    stats = []

    engine = Parallel()

    ##############################################################
    # ESTIMATE LOSS FOR EACH MEMBER BASED ON X_TRAIN PREDICTIONS #
    ##############################################################

    if X_train is not None and y_train is not None:
        if isinstance(model, (XGBRegressor)):
            model = BaggingRegressor(estimator=model, n_estimators=160)
            model.fit(X_train, y_train)
        elif isinstance(model, (XGBClassifier)):
            model = BaggingClassifier(estimator=model, n_estimators=160)
            model.fit(X_train, y_train)
        train_pred_jobs = _setup_pred_jobs(
            model, 
            X_train, 
            proba, 
            threshold
        )
        train_results = engine(train_pred_jobs)

        if proba:
            print("model is a classifier")  # for debug
            losses = 1 - np.array(
                [balanced_accuracy_score(y_train.astype(int), 
                pred.astype(int)) for pred in train_results]
            )

        else:
            print("model is a regressor")  # for debug
            losses = np.array([mean_absolute_error(y_train, y_inverse_transformer(pred)) for pred in train_results])

        weights = 1 / (losses + 1e-99)  # Avoid division by zero
        weights /= weights.sum()  # Normalize weights
    else:
        raise ValueError("X_train and y_train should be defined")
    
    ##########################################
    # COMPUTE PREDICTIONS BASED ON X_PREDICT #
    ##########################################

    for chunk in chunks:
        predict_pred_jobs = _setup_pred_jobs(model=model, 
                                             X=chunk, 
                                             proba=proba, 
                                             threshold=threshold)

        results = engine(predict_pred_jobs)
        print("results.shape: ", np.column_stack(results).shape)
        chunk_preds = pd.DataFrame(
            y_inverse_transformer(np.column_stack(results)),
            index=getattr(chunk, "index", None),
        )

        weights = None
        if weights is not None:
            lower = chunk_preds.apply(u.weighted_quantile, q=0.025, weights=weights, axis=1)
            upper =  chunk_preds.apply(u.weighted_quantile, q=0.975, weights=weights, axis=1)
        else:
            print("weights is None")
            lower = chunk_preds.quantile(q=0.025, axis=1)
            upper =  chunk_preds.quantile(q=0.975, axis=1)

        chunk_stats = pd.DataFrame(
            np.column_stack((lower, upper)), # stats we've calculated by chunk
            columns=['ci95_LL', 'ci95_UL'], # column names for the stats
            index = chunk.index # the index of the chunk, so we can align the results back to X_predict
        )
        stats.append(chunk_stats)
    
    output = pd.concat(stats, axis=0, ignore_index=False)
    return output

##################################
# SETUP PARALLEL PREDICTION JOBS #
##################################

def _setup_pred_jobs(model, X, proba, threshold):
    if isinstance(model, (XGBClassifier, XGBRegressor)):
        print("model is: ", model)
        print("model is XGBoost")
        n_estimators = u.xgboost_get_n_estimators(model)
        print("n_estimators: ", n_estimators)
        if n_estimators==None:
            raise ValueError("n_estimators not inferred correctly!")

        booster = model.get_booster()

        print("booster :", booster)
        pred_jobs = (
            delayed(u._predict_one_member)(
                i, 
                member=booster, 
                chunk=X, 
                proba=proba, 
                threshold=threshold    
            )
            for i in range(n_estimators)
        )
    else:
        print("model is: ", model)
        members, features_for_members = _flatten_metaensemble(model)
        print("features_for_members:", features_for_members)
        pred_jobs = (
            delayed(u._predict_one_member)(
                _,
                member=member,
                chunk=X.iloc[:, features_for_member],
                proba=proba, 
                threshold=threshold
            )
            for _, (features_for_member, member) in enumerate(zip(features_for_members, members))
        )
    return pred_jobs
        

#################################################################
# EXTRACT INDIVIDUAL MEMBERS FROM SKLEARN ENSEMBLE-BASED MODELS #
#################################################################

def _flatten_metaensemble(me):
    """Modified version to handle RandomForest's individual trees"""
    estimators = []
    features = []
    
    # Case 1: RandomForest - get all its trees
    if hasattr(me, 'estimators_') and isinstance(me, (RandomForestRegressor, RandomForestClassifier)):
        print("model inferred to be RF")
        n_features = me.n_features_in_
        estimators = me.estimators_
        features = [list(range(n_features))] * len(estimators)  # All trees use all features (but possibly different subsets)
    
    # Case 2: Bagging estimator (like BaggingRegressor with KNN)
    elif hasattr(me, 'estimators_') and hasattr(me, 'estimators_features_'):
        print(f"model inferred to be B-KNN and is actually {me}")
        estimators = me.estimators_
        features = me.estimators_features_
    
    # Case 3: Regular ensemble (Voting, Stacking)
    elif hasattr(me, 'estimators_'):
        print("Voting regressor or stacked")
        n_features = me.n_features_in_
        estimators = me.estimators_
        features = [list(range(n_features))] * len(estimators)
    
    # Case 4: Single estimator
    else:
        print("something else")
        estimators = [me]
        features = [list(range(getattr(me, 'n_features_in_', X_train.shape[1])))]  # Fallback to X_train shape
    
    return estimators, features


# Example Usage
if __name__ == "__main__":
    # Generate sample data
    from sklearn.datasets import make_regression
    from joblib import parallel_backend  # this is user-facing
    from abil.utils import ZeroInflatedRegressor, ZeroStratifiedKFold

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
    # cv = ZeroStratifiedKFold(n_splits=cv_splits)
    cv = KFold(n_splits=cv_splits)

    # Define model and method
    model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42).fit(
        X_train, y_train
    )

    vmodel = VotingRegressor(
        estimators=[
            (
                "rf",
                RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42),
            ),
            (
                "knn",
                BaggingRegressor(
                    KNeighborsRegressor(n_neighbors=10, weights="distance"),
                    n_estimators=50,
                ).fit(X_train, y_train),
            ),
        ]
    ).fit(X_train, y_train)

    mask = (y_train > 0).values
    zirmodel = ZeroInflatedRegressor(
        BaggingClassifier(
            KNeighborsClassifier(n_neighbors=10, weights="distance"), n_estimators=50
        ),        BaggingRegressor(

            KNeighborsRegressor(n_neighbors=10, weights="distance"), n_estimators=50
        ),
    ).fit(X_train, y_train)

    v_of_zirmodels = VotingRegressor(
        estimators=[
            ("rf-then-knn", zirmodel),
            (
                "knn-then-rf",
                ZeroInflatedRegressor(
                    BaggingClassifier(
                        KNeighborsClassifier(n_neighbors=10, weights="distance"),
                        n_estimators=50,
                    ).fit(X_train, mask),
                    RandomForestRegressor(
                        n_estimators=100, max_depth=4, random_state=2245
                    ).fit(X_train.iloc[mask, :], y_train.iloc[mask]),
                ),
            ),
        ]
    ).fit(X_train, y_train)

    zir_of_vmodels = ZeroInflatedRegressor(
        classifier=VotingClassifier(
            estimators=[
                ("rfc", RandomForestClassifier(
                    n_estimators=100, max_depth=4, random_state=2245
                )),
                ("bagnnc", BaggingClassifier(
                    KNeighborsClassifier(n_neighbors=10, weights="distance"),
                    n_estimators=50,
                )),
            ]
        ),
        regressor=VotingRegressor(
            estimators=[
                ("rfr", RandomForestRegressor(n_estimators=100, max_depth=4, random_state=2245)),
                ("bagnnr", BaggingRegressor(
                    KNeighborsRegressor(n_neighbors=10, weights="distance"),
                    n_estimators=50,
                ))
            ]
        ),
    ).fit(X_train, y_train)

    # this sets the backend type and number of jobs to use in the internal
    # Parallel() call.
    with parallel_backend("loky", n_jobs=16):
        results = estimate_prediction_quantiles(
            model, X_predict=X_predict, X_train=X_train, y_train=y_train, cv=cv
        )
    with parallel_backend("loky", n_jobs=16):
        vresults = estimate_prediction_quantiles(
            vmodel, X_predict=X_predict, X_train=X_train, y_train=y_train, cv=cv
        )
    with parallel_backend("loky", n_jobs=16):
        zirresults = estimate_prediction_quantiles(
            zirmodel, X_predict=X_predict, X_train=X_train, y_train=y_train, cv=cv
        )
    with parallel_backend("loky", n_jobs=16):
        vzir_results = estimate_prediction_quantiles(
            zir_of_vmodels, X_predict=X_predict, X_train=X_train, y_train=y_train, cv=cv
        )
    print("\n=== Training Summary Stats ===\n", results["train_stats"].head())
    print("\n=== Prediction Summary Stats ===\n", results["predict_stats"].head())

    print("\n=== Training Summary Stats ===\n", vzir_results["classifier_train_stats"].head())
    print("\n=== Prediction Summary Stats ===\n", vzir_results["classifier_predict_stats"].head())
    
    print("\n=== Training Summary Stats ===\n", vzir_results["regressor_train_stats"].head())
    print("\n=== Prediction Summary Stats ===\n", vzir_results["regressor_predict_stats"].head())
