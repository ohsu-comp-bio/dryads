
from .utils import get_square_gauss
from ..selection.pathway import PathwaySelect

import numpy as np
from abc import abstractmethod
import collections
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import pairwise


class BaseBayesianTransfer(BaseEstimator, ClassifierMixin):
    """Abstract class for transfer learning using variational Bayes.

    """

    def __init__(self,
                 path_keys=None, kernel='rbf', latent_features=5,
                 prec_distr=(1.0, 1.0), sigma_h=0.1, margin=1.0,
                 kern_gamma=-1.85, max_iter=200, stop_tol=1.0):
        self.path_keys = path_keys
        self.kernel = kernel
        self.R = latent_features

        self.prec_distr = prec_distr
        self.sigma_h = sigma_h
        self.margin = margin
        self.kern_gamma = kern_gamma

        self.max_iter = max_iter
        self.stop_tol = stop_tol

        self.expr_genes = None
        self.mut_genes = None

        self.feat_count = None
        self.weight_prior = None
        self.weight_mat = None

    def get_params(self, deep=True):
        return {'sigma_h': self.sigma_h,
                'prec_distr': self.prec_distr,
                'latent_features': self.R,
                'margin': self.margin,
                'kern_gamma': self.kern_gamma}

    def set_params(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def compute_kernels(self,
                        x_mat, y_mat=None, expr_genes=None, path_keys=None,
                        **kern_params):
        """Calculates the kernel between pairs of samples in two datasets.

        Args:
            x_mat (:obj:`np.array`, shape = [n_xsamps, n_features])
            y_mat (:obj:`np.array`, optional, shape = [n_ysamps, n_features])
                If only x_mat is given, y_mat defaults to x_mat, and the
                kernel is calculated between pairs of samples in x_mat.

        Returns:
            kern_mat (:obj:`np.array`, shape = [n_xsamps, n_ysamps])
                The type of kernel that is calculated is governed by the
                `self.kernel` attribute of the classifier.

        """

        if expr_genes is None:
            expr_genes = self.expr_genes

        xt_mat = PathwaySelect(path_keys).fit_transform(
            x_mat, expr_genes=expr_genes, **kern_params)

        if y_mat is not None:
            yt_mat = PathwaySelect(path_keys).fit_transform(
                y_mat, expr_genes=expr_genes, **kern_params)

        else:
            yt_mat = xt_mat

        if isinstance(self.kernel, collections.Callable):
            kern_mat = self.kernel(xt_mat, yt_mat)

        elif self.kernel == 'rbf':
            pair_dist = np.mean(pairwise.pairwise_distances(xt_mat))
            kern_mat = pairwise.rbf_kernel(xt_mat, yt_mat,
                                           gamma=pair_dist ** self.kern_gamma)

        elif self.kernel == 'linear':
            kern_mat = pairwise.linear_kernel(xt_mat, yt_mat)

        else:
            raise ValueError(
                "Unknown kernel " + str(self.kernel) + " specified!")

        return kern_mat

    @abstractmethod
    def init_output_mat(self, y_dict):
        """Inititalizes the matrix of predicted output scores.

        Args:
            y_dict (:obj:`dict` of :obj:`np.array`,
                    shape = [n_xsamps_i, n_phenos_i])
                The training output labels, given as a dictionary with an
                entry for each task consisting of an array with a row for each
                training sample and a column for each phenotype to predict.

        """

    def update_precision_priors(self,
                                precision_mat, variable_mat,
                                prec_alpha, prec_beta):
        """Updates posterior distributions of a matrix of precision priors.

        Performs an update step for the approximate posterior distributions
        of a matrix of gamma-distributed precision priors for a set of
        normally-distributed downstream variables.

        Args:
            precision_mat (dict): Current precision prior posteriors.
            variable_mat (dict): Current downstream variable posteriors.

        Returns:
            new_priors (dict): Updated precision priors.

        """
        new_priors = {'alpha': np.full(precision_mat['alpha'].shape,
                                       prec_alpha + 0.5),
                      'beta': (prec_beta
                               + 0.5 * get_square_gauss(variable_mat))}

        return new_priors

    @abstractmethod
    def get_pred_class_probs(self, pred_mu, pred_sigma):
        """Gets the predicted probability of falling into output classes."""

