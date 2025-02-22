
from ..base import StanClassifier
import numpy as np
from scipy.special import expit


class BaseLogistic(StanClassifier):
    """Bayesian logistic linear regression.

    Parameters:
        alpha (float): Regularization parameter for feature coefficients.
                       Lower values will put a higher penalty on coefficients
                       that are further from zero.
 
    """

    model_name = "BaseLogisticClassifier"

    def __init__(self, model_code, alpha=0.01):
        self.alpha = alpha
        super().__init__(model_code)

    def get_data_dict(self, omic, pheno, **fit_params):
        return {'N': omic.shape[0], 'G': omic.shape[1],
                'expr': omic, 'mut': np.array(pheno, dtype=np.int),
                'alpha': self.alpha}

    def calc_pred_labels(self, omic):
        var_means = self.get_var_means()
        return np.dot(omic, var_means['gn_wghts']) + var_means['intercept']

    def calc_pred_p(self, pred_labels):
        return expit(pred_labels)

