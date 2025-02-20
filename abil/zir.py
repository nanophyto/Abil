import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone, is_regressor, is_classifier

class ZeroInflatedRegressor(BaseEstimator, RegressorMixin):
    """
    A custom regressor to handle zero-inflated target variables.

    Combines a classifier to predict non-zero occurrences and a regressor for non-zero targets.
    """

    def __init__(self, classifier, regressor, threshold=0.5):
        """
        Initialize the regressor with a classifier and regressor.

        Parameters
        ----------
        classifier : estimator
            A classifier to predict non-zero values.
        regressor : estimator
            A regressor to predict non-zero targets.
        threshold : float
            The probability cutoff for predicting presence 

        """
        self.classifier = classifier
        self.regressor = regressor
        self.threshold = threshold

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training data.

        y : np.ndarray, 1-dimensional
            The target values.

        sample_weight : Optional[np.array], default=None
            Individual weights for each sample.

        Returns
        -------
        ZeroInflatedRegressor
            Fitted regressor.

        Raises
        ------
        ValueError
            If `classifier` is not a classifier or `regressor` is not a regressor.
        """
        if not is_classifier(self.classifier):
            raise ValueError(
                f"`classifier` has to be a classifier. Received instance of {type(self.classifier)} instead.")
        if not is_regressor(self.regressor):
            raise ValueError(f"`regressor` has to be a regressor. Received instance of {type(self.regressor)} instead.")

        # Ensure classifier_ is assigned
        self.classifier_ = clone(self.classifier)
        self.classifier_.fit(X, y != 0)

        # Ensure regressor_ is assigned
        self.regressor_ = clone(self.regressor)
        
        y_pred_proba = self.classifier_.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        non_zero_indices = np.where(y_pred == 1)[0]

        if non_zero_indices.size > 0:
            if isinstance(X, pd.DataFrame):
                self.regressor_.fit(
                    X.iloc[non_zero_indices] if isinstance(X, pd.DataFrame) else X[non_zero_indices],
                    y.iloc[non_zero_indices].values if isinstance(y, pd.Series) else y[non_zero_indices]
                )
            else:
                self.regressor_.fit(
                        X[non_zero_indices],
                        y[non_zero_indices],
                )
        else:
            print("All predictions are zero (!), skipping regressor fitting.")
        
        return self


    def predict(self, X):
        """
        Get predictions.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Samples to get predictions of.

        Returns
        -------
        y : np.ndarray, shape (n_samples,)
            The predicted values.
        """
        output = np.zeros(len(X))

        non_zero_indices = np.where(self.classifier_.predict(X) == 1)[0]

        if non_zero_indices.size > 0:
            if isinstance(X, pd.DataFrame):
                output[non_zero_indices] = self.regressor_.predict(X.iloc[non_zero_indices])
            else:
                output[non_zero_indices] = self.regressor_.predict(X[non_zero_indices])

        return output
