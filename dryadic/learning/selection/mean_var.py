
import numpy as np
from sklearn.base import TransformerMixin


class SelectMeanVar(TransformerMixin):

    def __init__(self, mean_perc=99, var_perc=99):
        self.mean_perc = mean_perc
        self.var_perc = var_perc
        self.mask_ = None

        return super().__init__()

    def fit(self, X, y=None, **fit_params):
        mean_vals = np.mean(X, axis=0)
        var_vals = np.var(X, axis=0)
        
        self.mask_ = mean_vals > np.percentile(mean_vals,
                                               100 - self.mean_perc)
        self.mask_ &= (var_vals > np.percentile(var_vals,
                                                100 - self.var_perc))

        return self

    def _get_support_mask(self):
        return self.mask_

    def transform(self, X, y=None, **fit_params):
        return np.array(X)[:, self._get_support_mask()]

    def get_params(self, deep=True):
        return {'mean_perc': self.mean_perc,
                'var_perc': self.var_perc}
 
    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

