from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

class UpsampledZeroStratifiedKFold:
    """
    Custom cross-validation generator with upsampling of zero instances for stratified folds.
    """

    def __init__(self, n_splits=3):
        """
        Initialize the stratified K-fold with upsampling.

        Parameters
        ----------
        n_splits : int, default=3
            Number of folds.
        """
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        """
        Generate train-test splits with upsampling of the minority class in the training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.
        y : array-like, shape (n_samples,)
            Target variable.
        groups : array-like, optional
            Group labels for the samples, used for group-based splitting. Not used in this method.

        Yields
        ------
        train_indices : np.ndarray
            Indices for the training set with upsampled minority class.
        test_indices : np.ndarray
            Indices for the test set.

        Notes
        -----
        - Converts `y` into a binary variable (`1` for non-zero values, `0` otherwise) for stratified sampling.
        - Upsamples the minority class in the training set to match the size of the majority class.
        - Uses `StratifiedKFold` for generating splits based on the binary target variable.
        """
        y_binary = np.where(y!=0, 1, 0)

        for rx, tx in StratifiedKFold(n_splits=self.n_splits).split(X,y_binary):
            nix = np.where(y_binary[rx]==0)[0]
            pix = np.where(y_binary[rx]==1)[0]
            pixu = np.random.choice(pix, size=nix.shape[0], replace=True)
            ix = np.append(nix, pixu)
            rxm = rx[ix]
            yield rxm, tx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
    
    
class ZeroStratifiedKFold:
    """
    Custom cross-validation generator to handle zero-inflated targets with stratification.
    """

    def __init__(self, n_splits=3):
        """
        Initialize the stratified K-fold.

        Parameters
        ----------
        n_splits : int, default=3
            Number of folds.
        """
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        """
        Generate train-test splits with upsampling of the minority class in the training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.
        y : array-like, shape (n_samples,)
            Target variable.
        groups : array-like, optional
            Group labels for the samples, used for group-based splitting. Not used in this method.

        Yields
        ------
        train_indices : np.ndarray
            Indices for the training set with upsampled minority class.
        test_indices : np.ndarray
            Indices for the test set.

        Notes
        -----
        - Converts `y` into a binary variable (`1` for non-zero values, `0` otherwise) for stratified sampling.
        - Upsamples the minority class in the training set to match the size of the majority class.
        - Uses `StratifiedKFold` for generating splits based on the binary target variable.
        """
        y_binary = np.where(y!=0, 1, 0)

        # Check if there are any zeros in the array
        if any(element == 0 for element in y_binary):
            for rx, tx in StratifiedKFold(n_splits=self.n_splits).split(X,y_binary):
                yield rx, tx
        else:
            for rx, tx in KFold(n_splits=self.n_splits).split(X,y_binary):
                yield rx, tx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

