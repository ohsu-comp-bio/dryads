
from ..base import *
import numpy as np
from scipy.stats import norm


class BaseTranscripts(StanClassifier):
    """Bayesian transcript feature multi-level Dirichlet linear regression.

    Parameters:
        alpha (float): Regularization parameter for feature coefficients.
                       Lower values will put a higher penalty on coefficients
                       that are further from zero.

    """

    model_name = "BaseTranscriptClassifier"

    def __init__(self, model_code, alpha=1e-4, gamma=np.exp(1)):
        self.alpha = alpha
        self.gamma = gamma

        super().__init__(model_code)

    def fit(self, X, y=None, expr_genes=None, **fit_params):
        self.genes = expr_genes

        super().fit(X, y, **fit_params)

    def get_data_dict(self, omic, pheno, **fit_params):
        return {'expr': omic, 'mut': np.array(pheno, dtype=np.int),
                'N': omic.shape[0], 'T': omic.shape[1], 'tx_indx': np.unique(
                    self.genes, return_counts=True)[1].tolist(),
                'G': len(set(self.genes)), 'alpha': self.alpha,
                'gamma': self.gamma}

    def calc_pred_labels(self, omic):
        var_means = self.get_var_means()
        pred_lbls = var_means['intercept']
        pred_lbls += np.dot(omic,
                            var_means['gn_wghts_use'] * var_means['tx_wghts'])

        return pred_lbls

    def calc_pred_p(self, pred_labels):
        return pred_labels

