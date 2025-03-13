import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import   StandardScaler
from abil.zero_stratified_kfold import  UpsampledZeroStratifiedKFold
from abil.log_grid_search import LogGridSearch
from abil.utils import example_data 
from abil.zir import ZeroInflatedRegressor
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.model_selection import GridSearchCV


X_train, X_predict, y = example_data("test", n_samples=1000, n_features=3, noise=0.1, train_to_predict_ratio=0.7, random_state=59)
X_predict["feature_1"]


numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, ["feature_1", "feature_2", "feature_3"])
            ])

cv = UpsampledZeroStratifiedKFold(n_splits=3)

reg_estimator = RandomForestRegressor(random_state=1, oob_score=True)

reg_pipe = Pipeline(steps=[('preprocessor', preprocessor),
            ('estimator', reg_estimator)])

reg = LogGridSearch(reg_pipe, verbose = 1, cv=cv, 
        param_grid={"regressor__estimator__n_estimators":[100]}, scoring='r2')

reg_grid_search = reg.transformed_fit(X_train, y, log="yes", predictors = ["feature_1", "feature_2", "feature_3"])
m2 = reg_grid_search.best_estimator_

clf_estimator = RandomForestClassifier(random_state=1, oob_score=True)

clf_pipe = Pipeline(steps=[('preprocessor', preprocessor),
            ('estimator', clf_estimator)])

clf = GridSearchCV(clf_pipe, verbose = 1, cv=cv, 
        param_grid={"estimator__n_estimators":[100]}, scoring='balanced_accuracy')

clf_grid_search = clf.fit(X_train, y>0)

m1 = clf_grid_search.best_estimator_


zirmodel = ZeroInflatedRegressor(m1, m2).fit(X_train, y)



print(zirmodel.__dict__)

transform =zirmodel.inverse_func

y = np.array([2, 4, 6, 8, 10]).reshape(-1, 1)  # Convert to 2D

print(transform(y))


from sklearn.preprocessing import FunctionTransformer

inverse_transform = getattr(
    zirmodel, "inverse_func", FunctionTransformer().inverse_func
)

print(inverse_transform(100))

