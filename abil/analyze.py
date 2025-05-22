from sklearn import metrics
from sklearn import inspection
from sklearn import base
import numpy


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

    base.check_array(X_test)
    base.check_array(X_train)

    n_test, n_features = X_test.shape
    n_train, _ = X_train.shape
    assert n_features == X_train.shape[1], (
        "features must be the same for both training and test data."
    )

    if not feature_weights:
        feature_weights = numpy.ones(n_features)
    elif feature_weights == "permutation":
        if model is None:
            raise ValueError(
                "Model must be provided if permutation feature importance is used"
            )
        feature_weight_kwargs.setdefault("n_jobs", -1)
        feature_weight_kwargs.setdefault("n_repeats", 10)
        feature_weights = inspection.permutation_importance(
            model, X_train, y_train, **feature_weight_kwargs
        ).importances_mean
        feature_weights /= feature_weights.sum()
    else:
        assert len(feature_weights) == n_features, (
            "weights must be provided for all features"
        )
        feature_weights = feature_weights / sum(feature_weights)

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
            X_test * feature_weights[None, :],
            X_train * feature_weights[None, :],
            metric=metric,
        )
        test_to_train_d_min = test_to_train_d.min(axis=1)
        test_to_train_i = test_to_train_d.argmin(axis=1)

        di_test = test_to_train_d_min / d_mean
        lpd_test = ((test_to_train_d / test_to_train_d.mean()) < cutpoint).sum(axis=1)

    else:
        # if we don't need local point density, this can be used
        test_to_train_i, test_to_train_d_min = metrics.pairwise_distances_argmin_min(
            X_test * feature_weights[None, :],
            X_train * feature_weights[None, :],
            metric=metric,
        )
        di_test = test_to_train_d_min / d_mean
        lpd_test = numpy.empty_like(di_test) * numpy.nan

    aoa = di_test >= cutpoint
    if return_all:
        return aoa, di_test, lpd_test, cutpoint, test_to_train_d

    return aoa

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
    
