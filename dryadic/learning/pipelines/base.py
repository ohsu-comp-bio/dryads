
"""Frameworks for applying machine learning algorithms to -omics datasets.

This file contains classes used to organize feature selection, normalization,
and prediction methods into robust pipelines that can be used to infer
phenotypic information from -omic datasets.

"""

from ..utilities.cross_validation import (
    cross_val_predict_omic, OmicShuffleSplit)

import numpy as np
from numbers import Number

from functools import reduce
from operator import mul
from copy import copy
from inspect import getargspec

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr


class PipelineError(Exception):
    pass


class OmicPipe(Pipeline):
    """Extracting phenotypic predictions from -omics dataset(s).

    Args:
        steps (list): A series of transformations and classifiers.
            An ordered list of feature selection, normalization, and
            classification/regression steps, the last of which produces
            feature predictions.

    """

    # the parameters that are to be tuned, with either statistical
    # distributions or iterables to be sampled from as values
    tune_priors = {}

    def __init__(self, steps, path_keys=None):
        super().__init__(steps)
        self.expr_genes = None
        self.fit_genes = None

        self.cur_tuning = dict(self.tune_priors)
        self.path_keys = path_keys
        self.tune_params_add = None
        self.fit_params_add = None

    def __str__(self):
        """Prints the tuned parameters of the pipeline."""
        param_str = type(self).__name__ + ' with '

        if self.tune_priors:
            param_list = self.get_params()
            param_str += reduce(
                lambda x, y: x + ', ' + y,
                [k + ': ' + '%s' % float('%.4g' % param_list[k])
                 if isinstance(param_list[k], Number)
                 else k + ': ' + str(param_list[k])
                 for k in self.cur_tuning.keys()]
                )
        else:
            param_str += 'no tuned parameters.'

        return param_str

    def _fit(self, X, y=None, **fit_params):
        self._validate_steps()

        step_names = [name for name, _ in self.steps]
        fit_params_steps = {name: {} for name, step in self.steps
                            if step is not None}

        self.expr_genes = X.columns.get_level_values(0).tolist()
        use_genes = X.columns.get_level_values(0).tolist()

        if 'fit' in fit_params_steps and self.fit_params_add:
            for pname, pval in self.fit_params_add.items():
                fit_params_steps['fit'][pname] = pval

        for pname, pval in fit_params.items():
            if '__' in pname:
                step, param = pname.split('__', maxsplit=1)
                fit_params_steps[step][param] = pval

            else:
                for step, transform in self.steps:
                    trans_args = getargspec(transform.fit)
 
                    # only adds the fitting argument if the fit method of the
                    # given step supports it
                    if pname in trans_args.args or trans_args.keywords:
                        fit_params_steps[step][pname] = pval

        Xt = X
        for name, transform in self.steps[:-1]:

            if transform:
                if 'expr_genes' in getargspec(transform.fit).args:
                    fit_params_steps[name]['expr_genes'] = use_genes

                if hasattr(transform, "fit_transform"):
                    Xt = transform.fit_transform(
                        Xt, y, **fit_params_steps[name])

                else:
                    Xt = transform.fit(
                        Xt, y, **fit_params_steps[name]).transform(Xt)

                if hasattr(transform, '_get_support_mask'):
                    gene_arr = np.array(use_genes).reshape(1, -1)
                    use_genes = transform.transform(
                        gene_arr).flatten().tolist()

        if self._final_estimator is None:
            final_params = {}

        else:
            final_params = fit_params_steps[self.steps[-1][0]]
            self.fit_genes = use_genes

            if 'expr_genes' in getargspec(self._final_estimator.fit).args:
                final_params['expr_genes'] = use_genes

        return Xt, final_params

    def predict_train(self,
                      cohort, lbl_type='prob',
                      include_samps=None, exclude_samps=None,
                      include_feats=None, exclude_feats=None):
        return self.predict_omic(
            cohort.train_data(None,
                              include_samps, exclude_samps,
                              include_feats, exclude_feats)[0],
            lbl_type
            )

    def predict_test(self,
                     cohort, lbl_type='prob',
                     include_samps=None, exclude_samps=None,
                     include_feats=None, exclude_feats=None):
        return self.predict_omic(
            cohort.test_data(None,
                             include_samps, exclude_samps,
                             include_feats, exclude_feats)[0],
            lbl_type
            )

    def predict_omic(self, omic_data):
        """Gets a vector of phenotype predictions for an -omic dataset."""
        return self.predict(omic_data)

    @classmethod
    def extra_fit_params(cls, cohort):
        fit_params = {}

        #if hasattr(cohort, 'path'):
        #    fit_params.update({'path_obj': cohort.path})

        return fit_params

    @classmethod
    def extra_tune_params(cls, cohort):
        return cls.extra_fit_params(cohort)

    def parse_preds(self, pred_omic):
        return pred_omic

    def score(self, X, y=None, sample_weight=None):
        """Get the accuracy of the classifier in predicting phenotype values.
        
        Used to ensure compatibility with cross-validation methods
        implemented in :module:`sklearn`.

        Args:
            X (array-like), shape = [n_samples, n_feats]
                A matrix of -omic values.

            y (array-like), shape = [n_samples, ]
                A vector of phenotype values.

            sample_weight (array-like), default = None
                If not None, how much weight to assign to the prediction for
                each sample when calculating the score.

        Returns:
            S (float): A score corresponding to prediction accuracy.
                The way this score is calculated is determined by the
                pipeline's `score_omic` method.

        """
        return self.score_omic(y, self.parse_preds(self.predict_omic(X)))

    def score_omic(self, actual_omic, pred_omic):
        """Scores the predictions for a set of phenotypes."""
        return self.score_pheno(actual_omic, pred_omic)

    def score_pheno(self, actual_pheno, pred_pheno):
        """Scores the predictions for a single phenotype."""
        raise NotImplementedError("An -omic pipeline used for prediction "
                                  "must implement the <score_pheno> method!")

    def tune_coh(self,
                 cohort, pheno,
                 tune_splits=2, test_count=8, parallel_jobs=16,
                 include_samps=None, exclude_samps=None,
                 include_feats=None, exclude_feats=None,
                 verbose=False):
        """Tunes the pipeline by sampling over the tuning parameters."""

        # checks if the classifier has parameters to be tuned, and how many
        # parameter combinations are possible
        if self.tune_priors:
            prior_counts = [len(x) if hasattr(x, '__len__') else float('Inf')
                            for x in self.cur_tuning.values()]
            max_tests = reduce(mul, prior_counts, 1)
            test_count = min(test_count, max_tests)

            train_omics, train_pheno = cohort.train_data(
                pheno,
                include_samps, exclude_samps,
                include_feats, exclude_feats
                )

            # get internal cross-validation splits in the training set and use
            # them to tune the classifier
            tune_cvs = OmicShuffleSplit(
                n_splits=tune_splits, test_size=0.2,
                random_state=(cohort.get_seed() ** 2) % 42949672
                )

            # samples parameter combinations and tests each one
            grid_test = RandomizedSearchCV(
                estimator=self, param_distributions=self.cur_tuning,
                n_iter=test_count, cv=tune_cvs, refit=False,
                n_jobs=parallel_jobs, pre_dispatch='n_jobs'
                )

            #TODO: figure out why passing extra_tune_params breaks in the new
            # scikit-learn code
            extra_params = self.extra_tune_params(cohort)
            grid_test.fit(X=train_omics, y=train_pheno, **extra_params)

            # finds the best parameter combination and updates the classifier
            tune_scores = (grid_test.cv_results_['mean_test_score']
                           - grid_test.cv_results_['std_test_score'])
            best_indx = tune_scores.argmax()

            best_params = grid_test.cv_results_['params'][best_indx]
            for par in best_params.keys() & extra_params.keys():
                del best_params[par]

            self.set_params(**best_params)
            if verbose:
                print(self)

        return self, grid_test.cv_results_

    def fit_coh(self,
                cohort, pheno,
                include_samps=None, exclude_samps=None,
                include_feats=None, exclude_feats=None):
        """Fits a classifier."""

        train_omics, train_pheno = cohort.train_data(
            pheno,
            include_samps, exclude_samps,
            include_feats, exclude_feats
            )
        self.fit_params_add = self.extra_fit_params(cohort)

        return self.fit(X=train_omics, y=train_pheno)

    def fit_transform_coh(self,
                          cohort, pheno=None,
                          include_samps=None, exclude_samps=None,
                          include_feats=None, exclude_feats=None):

        train_omics, train_pheno = cohort.train_data(
            pheno,
            include_samps, exclude_samps,
            include_feats, exclude_feats
            )
        self.fit_params_add = self.extra_fit_params(cohort)

        return self.fit_transform(X=train_omics, y=train_pheno)

    def eval_coh(self,
                 cohort, pheno, use_train=False,
                 include_samps=None, exclude_samps=None,
                 include_feats=None, exclude_feats=None):
        """Evaluate the performance of a classifier."""

        if use_train:
            test_omics, test_pheno = cohort.train_data(
                pheno,
                include_samps, exclude_samps,
                include_feats, exclude_feats
                )

        else:
            test_omics, test_pheno = cohort.test_data(
                pheno,
                include_samps, exclude_samps,
                include_feats, exclude_feats
                )

        return self.score(test_omics, test_pheno)

    def infer_coh(self,
                  cohort, pheno, force_test_samps=None, lbl_type='raw',
                  infer_splits=16, infer_folds=4, parallel_jobs=8,
                  include_samps=None, exclude_samps=None,
                  include_feats=None, exclude_feats=None):

        train_omics, train_pheno = cohort.train_data(
            pheno,
            include_samps, exclude_samps,
            include_feats, exclude_feats
            )

        return cross_val_predict_omic(
            estimator=self, X=train_omics, y=train_pheno,
            force_test_samps=force_test_samps, lbl_type=lbl_type,
            cv_fold=infer_folds, cv_count=infer_splits, n_jobs=parallel_jobs,
            fit_params=self.extra_fit_params(cohort),
            random_state=int(cohort.get_seed() ** 1.5) % 42949672,
            )

    def get_coef(self):
        """Get the fitted coefficient for each gene in the -omic dataset."""
        
        raise NotImplementedError("-omic pipelines must implement their own "
                                  "<get_coef> methods wherever possible!")


class PresencePipe(OmicPipe):
    """A class corresponding to pipelines which use continuous data to
       predict discrete outcomes.
    """

    def __init__(self, steps, path_keys=None):
        if not (hasattr(steps[-1][-1], 'predict_proba')
                or 'predict_proba' in steps[-1][-1].__class__.__dict__):

            raise PipelineError(
                "Variant pipelines must have a classification estimator "
                "with a 'predict_proba' method as their final step!"
                )

        super().__init__(steps, path_keys)

    def parse_preds(self, preds):
        if hasattr(self, 'classes_'):
            true_indx = [i for i, x in enumerate(self.classes_) if x]

            if len(true_indx) < 1:
                raise PipelineError("Classifier doesn't have a <True> class!")

            elif len(true_indx) > 1:
                raise PipelineError("Classifier has multiple <True> classes!")

        else:
            true_indx = [-1]

        if isinstance(preds, dict):
            new_preds = {k: np.array([scrs[true_indx[0]]
                                      for scrs in pred_list])
                         for k, pred_list in preds.items()}

        elif hasattr(preds[0], '__iter__'):
            new_preds = np.array([scrs[true_indx[0]] for scrs in preds])

        else:
            new_preds = preds

        return new_preds

    def predict_omic(self, omic_data, lbl_type='prob'):
        if lbl_type == 'prob':
            return self.predict_proba(omic_data)
        elif lbl_type == 'raw':
            return self.calc_pred_labels(omic_data)

        else:
            raise ValueError(
                "Unrecognized type of label `{}` !".format(lbl_type))

    @staticmethod
    def score_pheno(actual_pheno, pred_pheno):
        if len(pred_pheno.shape) != len(actual_pheno.shape):
            pred_pheno = pred_pheno.reshape(actual_pheno.shape)

        if actual_pheno.shape != pred_pheno.shape:
            raise PipelineError("This pipeline predicts phenotypes with "
                                "shape {} that do not conform to the "
                                "original phenotype shape {}!".format(
                                    pred_pheno.shape, actual_pheno.shape))

        if len(actual_pheno.shape) == 1:
            actual_pheno = actual_pheno.reshape(-1, 1)
            pred_pheno = pred_pheno.reshape(-1, 1)

        pheno_scores = [0.5 for _ in range(actual_pheno.shape[1])]
        for i in range(actual_pheno.shape[1]):

            if (len(np.unique(actual_pheno[:, i])) > 1
                    and len(np.unique(pred_pheno[:, i])) > 1):
                pheno_scores[i] = roc_auc_score(actual_pheno[:, i],
                                                pred_pheno[:, i])

        return min(pheno_scores)


class ValuePipe(OmicPipe):
    """A class corresponding to pipelines which use continuous data to
       predict continuous outcomes.
    """

    @staticmethod
    def score_pheno(actual_pheno, pred_pheno):
        if len(pred_pheno.shape) != len(actual_pheno.shape):
            pred_pheno = pred_pheno.reshape(actual_pheno.shape)

        if actual_pheno.shape != pred_pheno.shape:
            raise PipelineError("This pipeline predicts phenotypes with "
                                "shape {} that do not conform to the "
                                "original phenotype shape {}!".format(
                                    pred_pheno.shape, actual_pheno.shape))

        if len(actual_pheno.shape) == 1:
            actual_pheno = actual_pheno.reshape(-1, 1)
            pred_pheno = pred_pheno.reshape(-1, 1)

        pheno_scores = [0 for _ in range(actual_pheno.shape[1])]
        for i in range(actual_pheno.shape[1]):
            
            if np.var(actual_pheno[:, i]) > 0:
                if np.var(pred_pheno[:, i]) > 0:
                    pheno_scores[i] = pearsonr(actual_pheno[:, i],
                                               pred_pheno[:, i])[0]

        return min(pheno_scores)


class LinearPipe(OmicPipe):
    """An abstract class for classifiers implementing a linear separator.

    """

    def calc_pred_labels(self, X):
        return self.decision_function(X).reshape(-1)

    def get_coef(self):
        if self.fit_genes is None:
            raise PipelineError("Gene coefficients only available once "
                                "the pipeline has been fit!")

        return {gene: coef for gene, coef in
                zip(self.fit_genes, self.named_steps['fit'].coef_.flatten())}


class EnsemblePipe(OmicPipe):
    """An abstract class for classifiers made up of ensembles of separators.

    """

    def fit(self, X, y=None, **fit_params):
        self.effect_direct = [
            ((X.iloc[y.flatten(), i].mean() - X.iloc[~y.flatten(), i].mean())
             > 0)
            for i in range(X.shape[1])
            ]

        return super().fit(X, y, **fit_params)

    def get_coef(self):
        return {gene: coef * (2 * direct - 1) for gene, coef, direct in
                zip(self.genes, self.named_steps['fit'].feature_importances_,
                    self.effect_direct)}

