Permutation feature importance
==============================

Abil calculates permutation importance by measuring the reduction in a
model's prediction score when each original environmental predictor is
shuffled. The calculation operates on the full fitted pipeline and thus
reports importance for the original columns, including before scaling or
one-hot encoding.

After constructing an ``AbilPostProcessor`` with the training data and
configuration used for a model run:

.. code-block:: python

   summary, repeats = post.estimate_feature_importance(
       n_repeats=10, scoring="r2", n_jobs=1
   )

The default evaluates each configured model (and ``ens``) on held-out
cross-validation folds, refitting a clone of the saved estimator for each
fold. The summary reports mean and standard deviation across folds and
permutation repeats; the detailed table retains individual fold and repeat
results. Both tables are saved as CSV in ``posts/feature_importance/``.

For quicker exploratory analysis of the already-fitted model, use
``cv=False``. Such importances are in-sample rather than held-out.
To evaluate with spatially or temporally blocked folds, pass a corresponding
scikit-learn cross-validator and the group labels:

.. code-block:: python

   from sklearn.model_selection import GroupKFold

   summary, repeats = post.estimate_feature_importance(
       cv=GroupKFold(n_splits=5), groups=site_ids
   )

Here ``site_ids`` has one entry per training row, in the original training
order. For two-phase models, importance describes the combined predicted
concentration (occurrence and positive concentration together).

Negative importance is meaningful: permuting a predictor occasionally
improves the score. Highly correlated predictors can share importance.
Cross-validation uses the fitted model's selected hyperparameters without
re-running model selection in each fold; fully nested selection requires an
outer training workflow.

.. autofunction:: abil.analyze.permutation_feature_importance
