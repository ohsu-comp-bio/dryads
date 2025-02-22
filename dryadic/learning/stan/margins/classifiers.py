
from ..base import *
import numpy as np
from scipy.stats import norm, cauchy


class BaseMargin(StanClassifier):
    """Bayesian asymmetric overlap linear regression.

    Parameters:
        alpha (float): Regularization parameter for feature coefficients.
                       Lower values will put a higher penalty on coefficients
                       that are further from zero.

    """

    model_name = "BaseMarginClassifier"

    def __init__(self,
                 model_code,
                 wt_distr=(-1.0, 0.1), mut_distr=(1.0, 0.1), alpha=0.01):
        self.wt_distr = wt_distr
        self.mut_distr = mut_distr
        self.alpha = alpha

        super().__init__(model_code)

    def get_data_dict(self, omic, pheno, **fit_params):
        return {'Nw': np.sum(~pheno), 'Nm': np.sum(pheno), 'G': omic.shape[1],
                'expr_w': omic[~pheno], 'expr_m': omic[pheno],
                'wt_distr': self.wt_distr, 'mut_distr': self.mut_distr,
                'alpha': self.alpha}

    def calc_pred_labels(self, omic):
        var_means = self.get_var_means()
        return np.dot(omic, var_means['gn_wghts']) + var_means['intercept']


class GaussLabels(BaseMargin):

    def calc_pred_p(self, pred_labels):
        neg_logl = norm.logpdf(pred_labels, *self.wt_distr)
        pos_logl = norm.logpdf(pred_labels, *self.mut_distr)

        return 1 / (1 + np.exp(np.clip(neg_logl - pos_logl, -100, 100)))


class CauchyLabels(BaseMargin):

    def calc_pred_p(self, pred_labels):
        neg_logl = cauchy.logpdf(pred_labels, *self.wt_distr)
        pos_logl = cauchy.logpdf(pred_labels, *self.mut_distr)

        return 1 / (1 + np.exp(np.clip(neg_logl - pos_logl, -100, 100)))

