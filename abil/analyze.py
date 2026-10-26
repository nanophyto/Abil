from sklearn import metrics
from sklearn import inspection
from sklearn import base
from sklearn.base import clone
import numpy
import pandas as pd


def permutation_feature_importance(
    model, X, y, *, cv=None, scoring="r2", n_repeats=10,
    random_state=None, n_jobs=1, return_repeats=False, groups=None,
):
    """Estimate predictor importance through repeated score reductions.

    Parameters
    ----------
    model : fitted sklearn-compatible estimator
        The full model, including preprocessing and any ensemble components.
    X : array-like of shape (n_samples, n_features)
        Original, untransformed predictors. DataFrame column names are retained.
    y : array-like of shape (n_samples,)
        Observed targets for scoring.
    cv : cross-validator, optional
        If provided, clone and refit the estimator on each training fold and
        measure importance on its held-out fold. Without ``cv`` the supplied
        fitted model is evaluated on X and y.
    scoring : str or callable, default="r2"
        Scorer accepted by sklearn.inspection.permutation_importance.
    n_repeats : int, default=10
        Number of permutations per feature and evaluation fold.
    random_state : int, optional
        Seed for reproducible permutations.
    n_jobs : int, default=1
        Parallelism within each permutation-importance calculation.
    return_repeats : bool, default=False
        Also return one row per predictor, fold and permutation repeat.
    groups : array-like, optional
        Group labels supplied to ``cv.split`` when using grouped CV.

    Returns
    -------
    pandas.DataFrame or tuple of pandas.DataFrame
        Feature-level summary (``feature``, ``importance_mean``,
        ``importance_std``, ``n_folds``, ``n_repeats``), optionally followed
        by the repeat-level table (``feature``, ``fold``, ``repeat``,
        ``importance``). Importance is the *decrease* in the scoring metric;
        a negative value means that permuting improved the score.

    Notes
    -----
    When using cross-validation, estimator hyperparameters are retained from
    the supplied model; they are not retuned within folds.
    """
    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError("X must be a two-dimensional feature matrix")
    if len(y) != X.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    if groups is not None and len(groups) != len(y):
        raise ValueError("groups must contain one value per sample")

    names = list(X.columns) if hasattr(X, "columns") else [
        f"feature_{i}" for i in range(X.shape[1])
    ]
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    rng = numpy.random.RandomState(random_state)
    if cv is None:
        splits = [(None, numpy.arange(len(y)))]
    else:
        splits = cv.split(X, y, groups)

    def select(data, indices):
        return data.iloc[indices] if hasattr(data, "iloc") else numpy.asarray(data)[indices]

    records = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        estimator = model
        if train_idx is not None:
            estimator = clone(model).fit(select(X, train_idx), select(y, train_idx))
        result = inspection.permutation_importance(
            estimator, select(X, test_idx), select(y, test_idx),
            scoring=scoring, n_repeats=n_repeats,
            random_state=int(rng.randint(numpy.iinfo(numpy.int32).max)),
            n_jobs=n_jobs,
        )
        for i, name in enumerate(names):
            records.extend(
                (name, fold, repeat + 1, float(value))
                for repeat, value in enumerate(result.importances[i])
            )

    repeats = pd.DataFrame(records, columns=["feature", "fold", "repeat", "importance"])
    if repeats.empty:
        raise ValueError("cross-validation did not produce any evaluation folds")
    stats = repeats.groupby("feature", sort=False)["importance"].agg(
        importance_mean="mean", importance_std=lambda values: values.std(ddof=0)
    ).reset_index()
    stats["n_folds"] = repeats["fold"].nunique()
    stats["n_repeats"] = n_repeats
    return (stats, repeats) if return_repeats else stats



def area_of_applicability(
    X_test,
    X_train,
    y_train=None,
    model=None,
    cv=None,
    metric="euclidean",
    feature_weights="permutation",
    feature_weight_kwargs=None,
    threshold="tukey",
    return_all=False,
    chunk_size=10000
):
    """
    Estimate the area of applicability for the data using a strategy similar to Meyer & Pebesma 2022).

    This calculates the importance-weighted feature distances from test to train points,
    and then defines the "applicable" test sites as those closer than some threshold
    distance.

    Parameters
    ----------
    X_test  :   numpy.ndarray
        array of features to be used in the estimation of the area of applicability
    X_train :   numpy.ndarray
        the training features used to calibrate cutoffs for the area of applicability
    y_train :   numpy.ndarray
        the outcome values to estimate feature importance weights. Must be provided
        if the permutation feature importance is calculated.
    model   :   sklearn.BaseEstimator
        the model for which the feature importance will be calculated. Must be provided
        if the permutation feature importance is calculated.
    cv      : sklearn.BaseCrossValidator
        the crossvalidator to use on the input training data, in order to calculate the 
        within-sample extrapolation threshold
    metric  :   str (Default: 'euclidean')
        the name of the metric used to calculate feature-based distances.
    feature_weights : str or numpy.ndarray (Default: 'permutation')
        the name of the feature importance weighting strategy to be used. By default,
        scikit-learn's permutation feature importance is used. Pre-calculated
        feature importance scores can also be used. To ignore feature importance,
        set feature_weights=False.
    feature_weight_kwargs : dict()
        options to pass to the feature weight estimation function. By default, these
        are passed directly to sklearn.inspection.permutation_importance()
    threshold   :   str or float (Default: 'tukey')
        how to calculate the cutoff value to determine whether a model is applicable
        for a given test point. This cutoff is calculated within the training
        data, and applied to the test data.
        - 'tukey': use the tukey rule, setting the cutoff at 1.5 times the inter-quartile range (IQR) above the upper hinge (75th percentile) for the train data
        dissimilarity index.
        - 'mad': use a median absolute deviation rule, setting the cutoff at three times
        the median absolute deviation above the median train data dissimilarity index
        - float: if a value between zero and one is provided, then the cutoff is set at the
        percentile provided for the train data dissimilarity index.
    return_all: bool (Default: False)
        whether to return the dissimilarity index and density of train points near the test 
        point. Specifically, the dissimilarity index is the distance from test to train points in feature space, divided by the average distance between training points. The local density is the count of training datapoints whose feature distance is closer than the threshold value.

    Returns
    -------
        If return_local_density=False, the output is a numpy.ndarray of shape (n_training_samples, ) describing where a model
        might be considered "applicable" among the test samples.

        If return_local_density=True, then the output is a tuple of numpy arrays.
        The first element is the applicability mentioned above, the second is the 
        dissimilarity index for the test points, and the thord
        is the local density of training points near each test point.

        A value of 0 indicates the point is within the Area of Applicability, 
        while a value of 1 indicates the point is outside the Area of Applicability.
    """
    if feature_weight_kwargs is None:
        feature_weight_kwargs = dict()
    
    for features in (X_test, X_train):
        try:
            base.check_array(features, force_all_finite='allow-nan')
        except TypeError as exc:
            # scikit-learn >= 1.6 renamed force_all_finite.
            if 'force_all_finite' not in str(exc):
                raise
            base.check_array(features, ensure_all_finite='allow-nan')

    mask_test = numpy.asarray(~numpy.isnan(X_test).any(axis=1))
    mask_train = numpy.asarray(~numpy.isnan(X_train).any(axis=1))
    
    if y_train is not None:
        mask_train *= (~numpy.isnan(numpy.asarray(y_train)))

    n_test, n_features = X_test.shape
    n_train, _ = X_train.shape

    if n_features != X_train.shape[1]:
        raise ValueError(
        "features must be the same for both training and test data."
    )

    if mask_test.sum() <= 2:
        raise ValueError("at least two test samples must be not nan")
    if mask_train.sum() <= 2:
        raise ValueError("at least two train samples must be not nan")

    aoa_ = numpy.empty((n_test, ))
    di_test_ = numpy.empty((n_test, ))
    lpd_test_ = numpy.empty((n_test, ))

    if return_all:
        lpd_test_ = numpy.empty((n_test, ))
        test_to_train_d_ = numpy.empty((n_test, n_train))

    X_train = X_train[mask_train]

    # Only filter y_train if it exists
    if y_train is not None:
        y_train = numpy.asarray(y_train)[mask_train]


    if feature_weights is False or feature_weights is None:
        feature_weights = numpy.ones(n_features)
    elif isinstance(feature_weights, str):
        if feature_weights != "permutation":
            raise ValueError("unknown feature weighting strategy")
        if model is None or y_train is None:
            raise ValueError("model and y_train are required for permutation weights")
        options = dict(feature_weight_kwargs)
        options.setdefault("n_jobs", -1)
        options.setdefault("n_repeats", 10)
        options.pop("return_repeats", None)
        options.pop("cv", None)
        feature_weights = permutation_feature_importance(
            model, X_train, y_train, **options
        )["importance_mean"].to_numpy()
        # Negative importance is not a meaningful distance weight.
        feature_weights = numpy.maximum(feature_weights, 0)
    else:
        feature_weights = numpy.asarray(feature_weights, dtype=float)
        if feature_weights.shape != (n_features,):
            raise ValueError("weights must be provided for all features")
        if not numpy.all(numpy.isfinite(feature_weights)) or numpy.any(feature_weights < 0):
            raise ValueError("feature weights must be finite and non-negative")

    total_weight = feature_weights.sum()
    # Uniform weighting is well-defined when no permutation has positive importance.
    feature_weights = (feature_weights / total_weight if total_weight > 0
                       else numpy.full(n_features, 1 / n_features))

    train_distance = metrics.pairwise_distances(
        X_train * feature_weights[None, :], metric=metric
    )
    numpy.fill_diagonal(train_distance, train_distance.max())

    if cv is not None:
        d_mins = numpy.empty((X_train.shape[0],))
        mean_acc_num = 0
        mean_acc_den = 0
        for test_ix, train_ix in cv.split(X_train):
            hold_to_seen_d = train_distance[test_ix.reshape(-1,1), train_ix]
            d_mins[test_ix] = hold_to_seen_d.min(axis=1)
            mean_acc_num += hold_to_seen_d.sum()
            mean_acc_den += hold_to_seen_d.size
        d_mean = mean_acc_num/mean_acc_den    
    else:
        d_mins = train_distance.min(axis=1)
        numpy.fill_diagonal(train_distance, 0)
        d_mean = d_mean = train_distance[train_distance > 0].mean()

    di_train = d_mins / d_mean

    if threshold == "tukey":
        lo_hinge, hi_hinge = numpy.percentile(di_train, (0.25, 0.75))
        iqr = hi_hinge - lo_hinge
        cutpoint = iqr * 1.5 + hi_hinge
    elif threshold == "mad":
        median = numpy.median(di_train)
        mad = numpy.median(numpy.abs(di_train - median))
        cutpoint = median + 3 * mad
    elif (0 < threshold) & (threshold < 1):
        cutpoint = numpy.percentile(di_train, threshold)
    cutpoint = numpy.maximum(cutpoint, di_train.max())

    if return_all:
        test_to_train_d = metrics.pairwise_distances(
            X_test[mask_test] * feature_weights[None, :],
            # NOTE: X_train is already filtered from the previous step!
            X_train * feature_weights[None, :],
            metric=metric,
        )  

        # NOTE: This is gross, but I don't see how to work around this. 
        # I tried doing something along the lines of: 
        # ```{python}
        # test_to_train_d_[
        #     mask_test.reshape(-1,1)*mask_test.reshape(1,-1)
        # ] = test_to_train_d
        # ```
        # but you can't assign a 2d matrix *into* a 2d selection. 
        test_to_train_d_[:] = numpy.nan
        i_in_full = 0
        for i, is_null in enumerate(mask_test):
            if not is_null:
                test_to_train_d_[i, mask_train] = test_to_train_d[i_in_full]
                i_in_full+=1

        test_to_train_d_min = test_to_train_d.min(axis=1)
        test_to_train_i = test_to_train_d.argmin(axis=1)

        di_test = test_to_train_d_min / d_mean
        lpd_test = ((test_to_train_d / test_to_train_d.mean()) < cutpoint).sum(axis=1)

    else:
        # if we don't need local point density, this can be used
        test_to_train_i, test_to_train_d_min = metrics.pairwise_distances_argmin_min(
            X_test[mask_test] * feature_weights[None, :],
            # NOTE: X_train is already filtered from the previous step!
            X_train * feature_weights[None, :],
            metric=metric,
        )
        di_test = test_to_train_d_min / d_mean
        lpd_test = numpy.empty_like(di_test) * numpy.nan

    aoa = di_test >= cutpoint

    lpd_test_[mask_test] = lpd_test
    di_test_[mask_test] = di_test
    aoa_[mask_test] = aoa
    lpd_test_[~mask_test] = di_test_[~mask_test] = aoa_[~mask_test] = numpy.nan
    
    if return_all:
        return aoa_, di_test_, lpd_test_, cutpoint, test_to_train_d_
    else:
        return aoa_, di_test_, cutpoint

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import GridSearchCV

    # Generate X and y
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    # Add random latitude and longitude and set as MultiIndex
    latitudes = np.random.uniform(-90, 90, size=X.shape[0])
    longitudes = np.random.uniform(-180, 180, size=X.shape[0])
    X.index = pd.MultiIndex.from_tuples(zip(latitudes, longitudes), names=['Latitude', 'Longitude'])

    # The full X is the same as X_predict
    X_predict = X.copy()

    # Take a random subset and call this X_train and y_train
    sample_indices = np.random.choice(range(len(X)), size=100, replace=False) 
    X_train = X.iloc[sample_indices] 
    y_train = y[sample_indices]  

    # Initialize and fit the model
    reg = GridSearchCV(
        estimator = KNeighborsRegressor(),
        param_grid={"n_neighbors": [5, 8]},
        cv = KFold(n_splits=5))

    reg.fit(X_train, y_train)
    model = reg.best_estimator_   #this is what is currently loaded into post

    # Convert y_train to a pandas Series
    # in the pipeline y_train = df['species_name'], where df is a multi-index pandas dataframe
    y_train = pd.Series(y_train, index=X_train.index)

    aoa = area_of_applicability(
        X_test=X_predict,
        X_train=X_train,
        y_train=y_train,
        model=model
    )
    aoa2 = area_of_applicability(
        X_test=X_predict + 3,
        X_train=X_train,
        y_train=y_train,
        model=model,
        cv=reg.cv
    )
    ynan_train = y_train.copy()
    train_nans = X_train.sample(frac=.2).sample(frac=.2, axis=1)
    Xnan_train = X_train.copy()
    Xnan_train.loc[train_nans.index, train_nans.columns] = numpy.nan
    ynan_train[~ynan_train.index.isin(train_nans.index)].iloc[0] = numpy.nan
    Xnan_predict = X_predict.copy()
    test_nans = Xnan_predict.sample(frac=.2).sample(frac=.2,axis=1)
    Xnan_predict.loc[test_nans.index, test_nans.columns] = numpy.nan


    nantest = area_of_applicability(
        X_test=Xnan_predict,
        X_train=Xnan_train,
        y_train=ynan_train,
        model=model,
        return_all=True
    )

    if not numpy.isnan(
        nantest[0][X_predict.index.isin(test_nans.index)]
    ).all():
        raise ValueError("nan check failed, some nan values were dropped!")
    
