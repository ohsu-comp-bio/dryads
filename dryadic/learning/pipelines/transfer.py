
from .base import OmicPipe
import numpy as np
from inspect import getargspec
from copy import copy


class TransferPipe(OmicPipe):
    """A pipeline that transfers information between multiple datasets.

    """

    def _fit(self, X_dict, y_dict=None, **fit_params):
        if y_dict is None:
            y_dict = {lbl: None for lbl in X_dict}

        self._validate_steps()
        step_names = [name for name, _ in self.steps]

        use_genes = self.expr_genes
        fit_params_steps = {name: {} for name, step in self.steps
                            if step is not None}

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

        self.lbl_transforms = {lbl: [] for lbl in X_dict}
        Xt_dict = {lbl: None for lbl in X_dict}

        for lbl in X_dict:
            Xt = X_dict[lbl]
 
            for name, transform in self.steps[:-1]:
                if transform:
                    if 'expr_genes' in getargspec(transform.fit).args:
                        fit_params_steps[name]['expr_genes'] = use_genes[lbl]
 
                    if hasattr(transform, "fit_transform"):
                        Xt = transform.fit_transform(
                            Xt, y_dict[lbl], **fit_params_steps[name])
     
                    else:
                        Xt = transform.fit(
                            Xt, y_dict[lbl],
                            **fit_params_steps[name]
                            ).transform(Xt)

                    if hasattr(transform, '_get_support_mask'):
                        gene_arr = np.array(use_genes[lbl]).reshape(1, -1)
                        use_genes[lbl] = transform.transform(
                            gene_arr).flatten().tolist()

                self.lbl_transforms[lbl] += [(name, copy(transform))]

            Xt_dict[lbl] = Xt

        if self._final_estimator is None:
            final_params = {}
        else:
            final_params = fit_params_steps[self.steps[-1][0]]
 
            if 'expr_genes' in getargspec(self._final_estimator.fit).args:
                final_params['expr_genes'] = use_genes

        return Xt_dict, final_params

    def fit(self, X, y=None, **fit_params):
        """Fits the steps of the pipeline in turn."""

        self.expr_genes = {lbl: X_mat.columns.get_level_values(0)
                           for lbl, X_mat in X.items()}

        Xt_dict, final_params = self._fit(X, y, **fit_params)
        if self._final_estimator is not None:
            self._final_estimator.fit(Xt_dict, y, **final_params)

        return self

    def predict_proba(self, X):
        Xt_dict = X

        for lbl in X:
            for name, transform in self.lbl_transforms[lbl]:

                if transform is not None:
                    Xt_dict[lbl] = transform.transform(Xt_dict[lbl])

        return self.steps[-1][-1].predict_proba(Xt_dict)

    def eval_coh_each(self,
                      cohort, pheno, use_train=False,
                      include_samps=None, exclude_samps=None,
                      include_genes=None, exclude_genes=None):

        if use_train:
            test_omics, test_pheno = cohort.train_data(
                pheno,
                include_samps, exclude_samps,
                include_genes, exclude_genes
                )

        else:
            test_omics, test_pheno = cohort.test_data(
                pheno,
                include_samps, exclude_samps,
                include_genes, exclude_genes
                )

        return self.score_cohorts(test_omics, test_pheno)

    #@staticmethod
    #def parse_preds(preds):
    #    return {lbl: np.array(x).flatten() for lbl, x in preds.items()}

    def score_omic(self, actual_omic, pred_omic):
        """Parses and scores the predictions for a set of phenotypes."""
        return min(self.score_each(actual_omic, pred_omic).values())

    def score_cohorts(self, X, y=None, sample_weight=None):
        return self.score_each(y, self.predict_omic(X))

    def score_each(self, actual_omic, pred_omic):
        return {lbl: self.score_pheno(actual_omic[lbl], pred_omic[lbl])
                for lbl in actual_omic}


class MultiPipe(OmicPipe):
    """A pipeline for predicting multiple phenotypes at once.

    """

    def parse_preds(self, preds):
        return preds
    """
    def score_omic(self, actual_omic, pred_omic):
        return np.min(self.score_each(actual_omic, pred_omic))

    def score_each(self, actual_omic, pred_omic):
        return [self.score_pheno(act_omic, p_omic)
                for act_omic, p_omic in zip(actual_omic.transpose(),
                                            pred_omic.transpose())]
    """

