import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV

class LogGridSearch:
    """
    Perform grid search with optional logarithmic transformation of the target variable.

    Supports evaluating models with no transformation, log-transformation, or both.
    """

    def __init__(self, m, verbose, cv, param_grid, scoring, regions=None):
        """
        Initialize the LogGridSearch.

        Parameters
        ----------
        m : estimator
            Base model for grid search.
        verbose : int
            Verbosity level of grid search.
        cv : int or cross-validation generator
            Cross-validation strategy.
        param_grid : dict
            Grid of parameters to search.
        scoring : str or callable
            Scoring metric.
        regions : optional
            Additional regions information (if required).
        """
        self.m = m
        self.verbose = verbose
        self.cv = cv
        self.param_grid = param_grid
        self.scoring = scoring
        self.regions = regions
    
    def do_log(self, x):
        """
        Apply natural logarithm transformation to the input values.

        Parameters
        ----------
        x : array-like
            Input values.

        Returns
        -------
        y : array-like
            Log-transformed values.
        """
        y = np.log(x+1)
        return(y)
    
    def do_nothing(self, x):
        """
        Apply no transformation to the input values.

        Parameters
        ----------
        x : array-like
            Input values.

        Returns
        -------
        y : array-like
            Non-transformed values.
        """
        return(x)

    def do_exp(self, x):
        """
        Apply exponential transformation to the input values.

        Parameters
        ----------
        x : array-like
            Input values.

        Returns
        -------
        y : array-like
            Exponentially transformed values.
        """
        y = np.exp(x)-1
        return(y)
    
    def transformed_fit(self, X, y, log, predictors):
        """
        Perform grid search with optional log transformation on the target variable.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        y : pd.Series or np.ndarray
            Target variable.
        log : str
            Transformation mode: "yes" (log transform), "no" (no transform), or "both" (test both).
        predictors : list
            Predictors used for training (not directly used in the function).

        Returns
        -------
        GridSearchCV
            Fitted grid search instance for the best-performing model.

        Notes
        -----
        - Applies log transformation using `TransformedTargetRegressor` when `log="yes"`.
        - If `log="both"`, compares models with and without log transformation.
        - Uses `self.param_grid`, `self.scoring`, and `self.cv` for grid search.

        Raises
        ------
        ValueError
            If `log` is not "yes", "no", or "both".
        """

        if log=="yes":

            model = TransformedTargetRegressor(self.m, func = self.do_log, inverse_func=self.do_exp)
            grid_search = GridSearchCV(model, param_grid = self.param_grid, scoring=self.scoring, refit=True,
                            cv = self.cv, verbose = self.verbose, return_train_score=True, error_score=-1e99)
            grid_search.fit(X, y)
        
        elif log=="no":

            model = TransformedTargetRegressor(self.m, func = self.do_nothing, inverse_func=self.do_nothing)
            grid_search = GridSearchCV(model, param_grid = self.param_grid, scoring=self.scoring, refit=True,
                            cv = self.cv, verbose = self.verbose, return_train_score=True, error_score=-1e99)
            grid_search.fit(X, y)

        elif log =="both":

            normal_m = TransformedTargetRegressor(self.m, func = self.do_nothing, inverse_func=self.do_nothing)
            grid_search1 = GridSearchCV(normal_m, param_grid = self.param_grid, scoring=self.scoring, refit=True,
                            cv = self.cv, verbose = self.verbose, return_train_score=True, error_score=-1e99)
            grid_search1.fit(X, y)

            log_m = TransformedTargetRegressor(self.m, func = self.do_log, inverse_func=self.do_exp)
            grid_search2 = GridSearchCV(log_m, param_grid = self.param_grid, scoring=self.scoring, refit=True,
                            cv = self.cv, verbose = self.verbose, return_train_score=True, error_score=-1e99)
            grid_search2.fit(X, y)

            if (grid_search1.best_score_ > grid_search2.best_score_):
                grid_search = grid_search1
                best_transformation = "nolog"
                print("best = nolog")
        
            elif (grid_search1.best_score_ < grid_search2.best_score_):
                grid_search = grid_search2
                best_transformation = "log"
                print("best = log")
            else:
                print("same performance for both models")
                grid_search = grid_search1
                best_transformation = "nobest"
            grid_search.cv_results_['best_transformation'] = best_transformation

        else: 
            print("defined log invalid, pick from yes, no or both")

        return grid_search
