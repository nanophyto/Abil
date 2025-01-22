from sklearn import metrics
from sklearn import inspection
from sklearn import base
import numpy


def area_of_applicability(
    X_test,
    X_train,
    y_train=None,
    model=None,
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
        feature_weight_kwargs /= feature_weight_kwargs.sum()
    else:
        assert len(feature_weights) == n_features, (
            "weights must be provided for all features"
        )
        feature_weights = feature_weights / sum(feature_weights)

    train_distance = metrics.pairwise_distances(
        X_train * feature_weights[None, :], metric=metric
    )
    numpy.fill_diagonal(train_distance, train_distance.max())
    d_mins = train_distance.min(axis=1)
    numpy.fill_diagonal(train_distance, 0)
    d_mean = train_distance[train_distance > 0].mean()
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

    if return_local_density:
        test_to_train_d = metrics.pairwise_distances(
            X_test * feature_weights[None, :],
            X_train * feature_weights[None, :],
            metric=metric,
        )
        test_to_train_d_min = test_to_train_d.min(axis=1)
        test_to_train_i = test_to_train_d.argmin(axis=1)

        di_test = test_to_train_d_min / d_mean
        lpd_test = (di_test < cutpoint).sum(axis=1)

    else:
        # if we don't need local point density, this can be used
        test_to_train_i, test_to_train_d_min = metrics.pairwise_distances_argmin_min(
            X_test * feature_weights[None, :],
            X_train * feature_weights[None, :],
            metric=metric,
        )
        di_test = test_to_train_d_min / d_mean
        lpd_test = numpy.empty_like(di_test) * numpy.nan

    aoa = di_test <= cutpoint
    if return_all:
        return aoa, di_test, lpd_test

    return aoa
