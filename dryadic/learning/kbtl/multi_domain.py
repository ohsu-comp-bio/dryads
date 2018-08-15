
"""Using transfer learning to predict one or more labels in a single domain.

This module contains classes corresponding to particular versions of Bayesian
transfer learning algorithms for predicting binary labels in one or more
tasks. Learning is done within one domain using one or more kernels to map the
decision boundary to higher dimensions using pathway-based feature selection.
Optimizing over the set of unknown parameters and latent variables in each
model is done using variational inference, in which the posterior probability
of each is approximated analytically.

Authors: Michal Grzadkowski <grzadkow@ohsu.edu>
         Hannah Manning <manningh@ohsu.edu>

"""

from .base import BaseBayesianTransfer

import numpy as np
from scipy import stats
from sklearn.exceptions import NotFittedError

from abc import abstractmethod
from functools import reduce
from operator import add


class BaseMultiDomain(BaseBayesianTransfer):
    """Base class for transfer learning classifiers in multiple domains.

    Args:

    Attributes:

    """

    def __init__(self,
                 path_keys=None, kernel='rbf', latent_features=5,
                 prec_distr=(1.0, 1.0), sigma_h=0.1, margin=1.0,
                 kern_gamma=-1.85, max_iter=200, stop_tol=1.0):
        self.lambda_mats = None
        self.A_mats = None
        self.H_mats = None

        self.weight_prior = None
        self.weight_mat = None
        self.output_mats = None

        self.X_dict = None
        self.kernel_mats = None
        self.kkt_mats = None
        self.kern_sizes = None
        self.samp_counts = None

        super().__init__(path_keys, kernel, latent_features, prec_distr,
                         sigma_h, margin, kern_gamma, max_iter, stop_tol)

    def fit(self,
            X, y=None, expr_genes=None, path_keys=None, verbose=False,
            **fit_params):
        """Fits the classifier.

        Args:
            X (array-like of float), shape = [n_samples, n_features]
            y_list (array-like of bool): Known output labels.
            verbose (bool): How much information to print during fitting.
            fit_params (dict): Other parameters to control fitting process.

        Returns:
            self (MultiVariant): The fitted instance of the classifier.

        """
        if expr_genes is None:
            expr_genes = {lbl: None for lbl in X}
        if path_keys is None:
            path_keys = {lbl: None for lbl in X}

        self.X_dict = X
        self.expr_genes = expr_genes
        self.path_keys = path_keys

        # computes the kernel matrices and concatenates them, gets number of
        # training samples and total number of kernel features
        self.kernel_mats = {
            lbl: self.compute_kernels(xmat, expr_genes=expr_genes[lbl],
                                      path_keys=path_keys[lbl])
            for lbl, xmat in X.items()
            }

        self.kkt_mats = {lbl: kern_mat @ kern_mat.transpose()
                         for lbl, kern_mat in self.kernel_mats.items()}
        self.kern_sizes = {lbl: kern_mat.shape[0]
                           for lbl, kern_mat in self.kernel_mats.items()}
        self.samp_counts = {lbl: kern_mat.shape[1]
                              for lbl, kern_mat in self.kernel_mats.items()}

        # makes sure training labels are of the correct format
        if not isinstance(y, dict):
            raise TypeError("Output labels must be given as a dictionary "
                            "with an entry for each task!")

        if not all(all(y in [True, False] for y in yvals)
                   for yvals in y.values()):
            raise TypeError("Output label values must all be boolean!")

        y_dict = {lbl: np.array(yvals, dtype=np.float) * 2 - 1
                  for lbl, yvals in y.items()}
        y_dict = {
            lbl: yvals.reshape(-1, 1) if len(yvals.shape) == 1 else yvals
            for lbl, yvals in y_dict.items()
            }

        feat_counts = {y.shape[1] for y in y_dict.values()}
        if len(feat_counts) > 1:
            raise ValueError("Each task must involve the same number of "
                             "output features!")

        # initializes matrix of posterior distributions of precision priors
        # for the projection matrix
        self.lambda_mats = {
            lbl: {'alpha': (np.zeros((kern_size, self.R))
                            + self.prec_distr[0] + 0.5),
                  'beta': (np.zeros((kern_size, self.R))
                           + self.prec_distr[1])}
            for lbl, kern_size in self.kern_sizes.items()
            }

        self.feat_count = tuple(feat_counts)[0]
        self.output_mats = self.init_output_mat(y_dict)

        # initializes posteriors of precision priors for coupled
        # classification matrices
        self.weight_prior = {
            'alpha': np.zeros((self.R + 1, self.feat_count)) + 1.5,
            'beta': np.zeros((self.R + 1, self.feat_count)) + 1.0
            }

        self.weight_mat = {
            'mu': np.vstack((np.zeros((1, self.feat_count)),
                             np.random.randn(self.R, self.feat_count))),
            'sigma': np.tile(np.eye(self.R + 1), (self.feat_count, 1, 1))
            }

        self.A_mats = {lbl: {'mu': np.random.randn(kern_size, self.R),
                             'sigma': np.array(np.eye(kern_size)[..., None]
                                               * ([1] * self.R)).transpose()}
                       for lbl, kern_size in self.kern_sizes.items()}

        self.H_mats = {lbl: {'mu': np.random.randn(self.R, samp_count),
                             'sigma': np.eye(self.R)}
                       for lbl, samp_count in self.samp_counts.items()}


        # proceeds with inference using variational Bayes for the given
        # number of iterations
        cur_iter = 1
        old_log_like = float('-inf')
        log_like_stop = False

        while cur_iter <= 10 and not log_like_stop:
            new_lambdas = {
                lbl: self.update_precision_priors(
                    self.lambda_mats[lbl], self.A_mats[lbl], *self.prec_distr)
                for lbl in X
                }

            new_projs = self.update_projection(new_lambdas)
            new_latents = self.update_latent(new_projs, y_dict)

            new_weight_prior = self.update_precision_priors(
                self.weight_prior, self.weight_mat, *self.prec_distr)
            new_weights = self.update_weights(new_weight_prior, new_latents)
            new_outputs = self.update_output(new_latents, new_weights,
                                             y_dict)

            self.lambda_mats = new_lambdas
            self.A_mats = new_projs
            self.H_mats = new_latents

            self.weight_prior = new_weight_prior
            self.weight_mat = new_weights
            self.output_mats = new_outputs

            # every ten update iterations, check the likelihood of the model
            # given current latent variable distributions
            if (cur_iter % 10) == 0:
                cur_log_like = self.log_likelihood(y_dict)

                if (cur_log_like < (old_log_like + self.stop_tol)
                        and cur_iter > 10):
                    log_like_stop = True

                else:
                    old_log_like = cur_log_like

                    if verbose:
                        print('Iteration {}: {}'.format(
                            cur_iter, cur_log_like))

            cur_iter += 1

        return self

    def predict_labels(self, X):
        kern_dists = {
            lbl: self.compute_kernels(x_mat=self.X_dict[lbl], y_mat=X[lbl],
                                      expr_genes=self.expr_genes[lbl])
            for lbl in X
            }

        h_mus = {lbl: self.A_mats[lbl]['mu'].transpose() @ kern_dists[lbl]
                 for lbl in X}
        f_mus = {lbl: np.zeros((X[lbl].shape[0], self.feat_count))
                 for lbl in X}
        f_sigmas = {lbl: np.zeros((X[lbl].shape[0], self.feat_count))
                    for lbl in X}

        for lbl in X:
            for i in range(self.feat_count):
                f_mus[lbl][:, i] = np.dot(
                    np.vstack((np.ones(X[lbl].shape[0]),
                               h_mus[lbl])).transpose(),
                    self.weight_mat['mu'][:, i]
                    )
                
                f_sigmas[lbl][:, i] = 1.0 + np.diag(
                    np.dot(
                        np.dot(np.vstack((np.ones(X[lbl].shape[0]),
                                          h_mus[lbl])).transpose(),
                               self.weight_mat['sigma'][i, :, :]),
                        np.vstack((np.ones(X[lbl].shape[0]), h_mus[lbl]))
                        )
                    )

        return f_mus, f_sigmas

    def predict_proba(self, X):
        f_mus, f_sigmas = self.predict_labels(X)
        pred_dict = {lbl: np.zeros((X[lbl].shape[0], self.feat_count))
                     for lbl in X}

        for lbl in X:
            for i in range(self.feat_count):
                pred_p, pred_n = self.get_pred_class_probs(
                    f_mus[lbl][:, i], f_sigmas[lbl][:, i])
                pred_dict[lbl][:, i] = pred_p / (pred_p + pred_n)

        return pred_dict

    @abstractmethod
    def get_pred_class_probs(self, pred_mu, pred_sigma):
        """Gets the predicted probability of falling into output classes."""

    @abstractmethod
    def init_output_mat(self, y_list):
        """Initialize posterior distributions of the output predictions."""

    def update_projection(self, new_priors):
        """Updates posterior distributions of projection matrices.

        Args:

        Returns:

        """
        new_variables = {
            lbl: {'mu': np.zeros(self.A_mats[lbl]['mu'].shape),
                  'sigma': np.zeros(self.A_mats[lbl]['sigma'].shape)}
            for lbl in new_priors
            }

        prior_expects = {
            lbl: (prior_mat['alpha'] / prior_mat['beta']).transpose().tolist()
            for lbl, prior_mat in new_priors.items()
            }

        for lbl in new_priors:
            for i in range(self.R):
                new_variables[lbl]['sigma'][i, :, :] = np.linalg.inv(
                    np.diag(prior_expects[lbl][i])
                    + (self.kkt_mats[lbl] / self.sigma_h)
                    )
 
                new_variables[lbl]['mu'][:, i] = np.dot(
                    new_variables[lbl]['sigma'][i, :, :],
                    np.dot(self.kernel_mats[lbl],
                           self.H_mats[lbl]['mu'][i, :].transpose())
                    / self.sigma_h
                    )

        return new_variables

    def update_latent(self, variable_mats, y_dict):
        """Updates latent feature matrix.

        Args:

        Returns:

        """
        new_latents = {
            lbl: {k: np.zeros(hmat.shape) for k, hmat in hdict.items()}
            for lbl, hdict in self.H_mats.items()
            }

        for lbl, var_mat in variable_mats.items():
            new_latents[lbl]['sigma'] = np.linalg.inv(
                np.diag([self.sigma_h ** -1 for _ in range(self.R)])
                + reduce(lambda x, y: x + y,
                         [np.outer(self.weight_mat['mu'][1:, i],
                                   self.weight_mat['mu'][1:, i])
                          + self.weight_mat['sigma'][i][1:, 1:]
                          for i in range(self.feat_count)])
                )

            new_latents[lbl]['mu'] = np.dot(
                new_latents[lbl]['sigma'],
                np.dot(variable_mats[lbl]['mu'].transpose(),
                       self.kernel_mats[lbl]) / self.sigma_h
                + reduce(lambda x, y: x + y,
                         [np.outer(self.weight_mat['mu'][1:, i],
                                   self.output_mats[lbl]['mu'][:, i])
                          - np.repeat(a=np.array([
                              [x * self.weight_mat['mu'][0, i] + y
                               for x, y in zip(
                                   self.weight_mat['mu'][1:, i],
                                   self.weight_mat['sigma'][i, 1:, 0])]
                            ]), repeats=self.samp_counts[lbl], axis=0
                          ).transpose()
                          for i in range(self.feat_count)])
                )

        return new_latents

    def update_weights(self, weight_priors, latent_mats):
        """Update the binary classification weights.

        Args:

        Returns:

        """
        new_weights = {'mu': np.zeros(weight_priors['alpha'].shape),
                       'sigma': np.zeros((self.feat_count,
                                          self.R + 1, self.R + 1))}

        h_sum = reduce(add,
                       [np.sum(latent_mat['mu'], axis=1)
                        for latent_mat in latent_mats.values()])

        hht_mat = reduce(add,
                         [(latent_mat['mu'] @ latent_mat['mu'].transpose()
                          + latent_mat['sigma'] * self.samp_counts[lbl])
                          for lbl, latent_mat in latent_mats.items()])

        for i in range(self.feat_count):
            new_weights['sigma'][i, 0, 0] = (
                weight_priors['alpha'][0, i] / weight_priors['beta'][0, i]
                + sum(self.samp_counts.values())
                )

            new_weights['sigma'][i, 1:, 0] = h_sum
            new_weights['sigma'][i, 0, 1:] = h_sum

            new_weights['sigma'][i, 1:, 1:] = (
                hht_mat + np.diag(weight_priors['alpha'][1:, i]
                                  / weight_priors['beta'][1:, i])
                )
            new_weights['sigma'][i, :, :] = np.linalg.inv(
                new_weights['sigma'][i, :, :])

            new_weights['mu'][:, i] = np.dot(
                new_weights['sigma'][i, :, :],
                reduce(add, [np.dot(np.vstack([np.ones(self.samp_counts[lbl]),
                                               latent_mat['mu']]),
                                    self.output_mats[lbl]['mu'][:, i])
                             for lbl, latent_mat in latent_mats.items()])
                )

        return new_weights

    @abstractmethod
    def update_output(self, latent_mat, weight_mat, y_list):
        """Update the predicted output labels."""

    @abstractmethod
    def get_y_logl(self, y_list):
        """Computes the log-likelihood of a set of output labels given
           the current model state.
        """

    def log_likelihood(self, y_dict):
        """Computes the log-likelihood of the current model state.

        Args:

        Returns:

        """
        if self.lambda_mats is None:
            raise NotFittedError(
                "Can't compute model likelihood before fitting!")

        # precision prior distribution given precision hyper-parameters
        prec_distr = stats.gamma(a=self.prec_distr[0],
                                 scale=self.prec_distr[1] ** -1.0)

        # likelihood of projection matrix precision priors given
        # precision hyper-parameters
        lambda_logl = sum(
            np.sum(prec_distr.logpdf(lambda_mat['alpha']
                                     / lambda_mat['beta']))
            for lambda_mat in self.lambda_mats.values()
            )

        # likelihood of projection matrix values given their precision priors
        a_logl = sum(
            np.sum(stats.norm(
                loc=0, scale=(self.lambda_mats[lbl]['beta']
                              / self.lambda_mats[lbl]['alpha'])
                ).logpdf(self.A_mats[lbl]['mu']))
            for lbl in y_dict
            )

        # likelihood of latent feature matrix given kernel matrix,
        # projection matrix, and standard deviation hyper-parameter
        h_logl = sum(
            np.sum(stats.norm(
                loc=(self.A_mats[lbl]['mu'].transpose()
                     @ self.kernel_mats[lbl]),
                scale=self.sigma_h
                ).logpdf(self.H_mats[lbl]['mu']))
            for lbl in y_dict
            )

        # likelihood of bias parameter precision priors given
        # precision hyper-parameters
        weight_prior_logl = np.sum(prec_distr.logpdf(
            self.weight_prior['alpha'] / self.weight_prior['beta']))

        # likelihood of bias parameters given their precision priors
        weight_logl = np.sum(stats.norm(
            loc=0,
            scale=self.weight_prior['beta'] / self.weight_prior['alpha']
            ).logpdf(self.weight_mat['mu']))

        # likelihood of predicted outputs given latent features, bias
        # parameters, and latent feature weight parameters
        f_logl = sum(
            np.sum(stats.norm(
                loc=(self.weight_mat['mu'][1:, :].transpose()
                     @ self.H_mats[lbl]['mu']
                     + np.vstack(self.weight_mat['mu'][0, :])).transpose(),
                scale=1
                ).logpdf(self.output_mats[lbl]['mu']))
            for lbl in y_dict
            )

        # likelihood of actual output labels given class separation margin
        # and predicted output labels
        return (lambda_logl + a_logl + h_logl + weight_prior_logl
                + weight_logl + f_logl + self.get_y_logl(y_dict))

    def get_output_distr(self, stat_list):
        """Gets the cumulative PDF of the output labels.

        Returns:
            out_probs (list): For each task, a list of (x, prob) pairs
        """
        out_probs = [[np.zeros(500) for _ in stat_list]
                     for _ in range(self.task_count)]

        # for each task, get the posterior distribution for each predicted
        # output label
        for i in range(self.task_count):
            distr_vec = [
                stats.norm(loc=mu, scale=sigma) for mu, sigma in
                zip(self.output_mat['mu'][i, :],
                    self.output_mat['sigma'][i, :])
                ]

            # calculate the range of possible predicted output values
            min_range = min(distr.ppf(0.01) for distr in distr_vec)
            max_range = max(distr.ppf(0.99) for distr in distr_vec)
            x_vals = np.linspace(min_range, max_range, 1000)

            # calculate the cumulative probability density function across all
            # label distributions at each possible predicted value
            for j, stat in enumerate(stat_list):
                out_probs[i][j] = [
                    (x,
                     np.mean([distr.pdf(x) for distr, st in
                              zip(distr_vec, stat) if st])
                     ) for x in x_vals
                    ]

        return out_probs

    def get_path_prior(self):
        proj_max = np.apply_along_axis(lambda x: max(abs(x)), 1,
                                       self.A_mat['mu'])
        proj_list = {}

        for i, pk in enumerate(self.path_keys):
            start_indx = self.sample_count * i
            end_indx = self.sample_count * (i + 1)
            proj_list[pk] = proj_max[start_indx:end_indx]

        return proj_list


class MultiDomain(BaseMultiDomain):

    def get_pred_class_probs(self, pred_mu, pred_sigma):
        pred_p = 1 - stats.norm.cdf((self.margin - pred_mu) / pred_sigma)
        pred_n = stats.norm.cdf((-self.margin - pred_mu) / pred_sigma)

        return pred_p, pred_n

    def init_output_mat(self, y_dict):
        """Initialize posterior distributions of the output predictions."""
        output_mats = {lbl: {'mu': abs(np.random.randn(*yvals.shape)),
                             'sigma': np.ones(yvals.shape)}
                       for lbl, yvals in y_dict.items()}

        for lbl, yvals in y_dict.items():
            output_mats[lbl]['mu'] += self.margin
            output_mats[lbl]['mu'] *= np.sign(yvals)

        return output_mats

    def update_output(self, latent_mats, weight_mat, y_dict):
        """Update the predicted output labels."""
        new_output = {
            lbl: {k: np.zeros(omat.shape) for k, omat in output_mat.items()}
            for lbl, output_mat in self.output_mats.items()
            }

        # gets the margin boundaries for each output label class
        lu_list = {lbl: {
            'lower': np.vectorize(
                lambda y: -1e40 if y <= 0 else self.margin)(yvals),
            'upper': np.vectorize(
                lambda y: 1e40 if y >= 0 else -self.margin)(yvals)
            } for lbl, yvals in y_dict.items()
            }

        for lbl, latent_mat in latent_mats.items():
            for i in range(self.feat_count):
                f_raw = np.dot(weight_mat['mu'][1:, i], latent_mat['mu'])
                f_raw += weight_mat['mu'][0, i]
 
                alpha_norm = (lu_list[lbl]['lower'][:, i] - f_raw).tolist()
                beta_norm = (lu_list[lbl]['upper'][:, i] - f_raw).tolist()
                norm_factor = [stats.norm.cdf(b) - stats.norm.cdf(a)
                               if a != b else 1.0
                               for a, b in zip(alpha_norm, beta_norm)]
                norm_factor = [1.0 if x == 0 else x for x in norm_factor]

                new_output[lbl]['mu'][:, i] = [
                    f + ((stats.norm.pdf(a) - stats.norm.pdf(b)) / n)
                    for a, b, n, f in
                    zip(alpha_norm, beta_norm, norm_factor, f_raw.tolist())
                    ]
                new_output[lbl]['sigma'][:, i] = [
                    1.0 + (a * stats.norm.pdf(a) - b * stats.norm.pdf(b)) / n
                    - ((stats.norm.pdf(a) - stats.norm.pdf(b)) ** 2) / n ** 2
                    for a, b, n in zip(alpha_norm, beta_norm, norm_factor)
                    ]

        return new_output

    def get_y_logl(self, y_dict):
        return sum(
            np.sum(stats.norm(
                loc=self.output_mats[lbl]['mu'] * np.vstack(y_dict[lbl]),
                scale=self.output_mats[lbl]['sigma']
                ).logsf(1))
            for lbl in y_dict
            )


class MultiDomainAsym(BaseMultiDomain):
    """A multi-task transfer learning classifier with asymmetric margins.

    Args:

    Attributes:

    """

    def __init__(self,
                 path_keys=None, kernel='rbf', latent_features=5,
                 prec_distr=(1.0, 1.0), sigma_h=0.1, margin=5.0/3,
                 max_iter=500, stop_tol=1.0):
        super(MultiVariantAsym, self).__init__(
            path_keys=path_keys, kernel=kernel,
            latent_features=latent_features, prec_distr=prec_distr,
            sigma_h=sigma_h, margin=margin,
            max_iter=max_iter, stop_tol=stop_tol)

    def get_pred_class_probs(self, pred_mu, pred_sigma):
        pred_p = stats.norm(loc=self.margin, scale=self.margin ** -1.0)\
            .pdf(pred_mu)
        pred_n = stats.norm(loc=0.0, scale=1.0).pdf(pred_mu)

        return pred_p, pred_n

    def init_output_mat(self, y_list):
        """Initialize posterior distributions of the output predictions."""
        output_mat = {
            'mu': np.array([[stats.norm.rvs(self.margin, self.margin ** -1.0)
                             if y == 1.0
                             else stats.norm.rvs(-2.0, 1.0)
                             for y in y_vec] for y_vec in y_list]),
            'sigma': np.array([[self.margin ** -1.0 if y == 1.0 else 3.0
                                for y in y_vec] for y_vec in y_list])
            }

        return output_mat

    def get_lu_list(self, y_list):
        return [{'lower': np.array([-1e40 if i <= 0
                                    else self.margin - self.margin ** -0.5
                                    for i in y]),
                 'upper': np.array([1e40 if i >= 0 else 2.0
                                    for i in y])}
                for y in y_list]

    def update_latent(self, variable_mat, weight_mat, output_mat, y_list):
        """Updates latent feature matrix.

        Args:

        Returns:

        """
        new_latent = {k: np.zeros(self.H_mat[k].shape) for k in self.H_mat}
        y_sigma = [np.array([self.margin ** -1.0 if i == 1.0 else 4.0
                             for i in y])
                   for y in y_list]

        new_latent['sigma'] = np.linalg.inv(
            np.diag([self.sigma_h ** -1.0 for _ in range(self.R)])
            + reduce(lambda x, y: x + y,
                     [(np.outer(weight_mat['mu'][1:, i],
                                weight_mat['mu'][1:, i])
                       + weight_mat['sigma'][i][1:, 1:])
                      / (np.prod(y_sigma[0]) ** (1.0 / self.sample_count))
                      for i in range(self.task_count)])
            )

        new_latent['mu'] = np.dot(
            new_latent['sigma'],
            np.dot(variable_mat['mu'].transpose(),
                   self.kernel_mat) / self.sigma_h
            + reduce(
                lambda x, y: x + y,
                [(np.outer(weight_mat['mu'][1:, i], output_mat['mu'][i, :])
                  - np.repeat(a=np.array([
                    [x * weight_mat['mu'][0, i] + y for x, y in
                     zip(weight_mat['mu'][1:, i],
                         weight_mat['sigma'][i, 1:, 0])]]
                    ), repeats=self.sample_count, axis=0).transpose())
                 / y_sigma[i]
                 for i in range(self.task_count)]
                )
            )

        return new_latent

    def update_output(self, latent_mat, weight_mat, y_list):
        """Update the predicted output labels."""
        new_output = {k: np.zeros(self.output_mat[k].shape)
                      for k in self.output_mat}

        for i in range(self.task_count):
            f_raw = np.dot(np.tile(weight_mat['mu'][1:, i], (1, 1)),
                           latent_mat['mu']) + weight_mat['mu'][0, i]

            neg_indx = np.array(y_list[i]) == -1.0
            pos_indx = np.array(y_list[i]) == 1.0

            adj_val = (np.percentile(f_raw[0, neg_indx],
                                    stats.norm.cdf(1.0) * 100)
                       - 1.0)

            if adj_val > 0:
                new_output['mu'][i, neg_indx] = f_raw[0, neg_indx] - adj_val
            else:
                new_output['mu'][i, neg_indx] = f_raw[0, neg_indx]
            new_output['sigma'][i, neg_indx] = 1.0

            new_output['mu'][i, pos_indx] = (
                ((f_raw[0, pos_indx] - np.mean(f_raw[0, pos_indx]))
                 / ((np.var(f_raw[0, pos_indx]) ** 0.5) * self.margin))
                + self.margin
                )
            new_output['sigma'][i, pos_indx] = self.margin ** -1.0

        return new_output

    def get_y_logl(self, y_list):
        y_hat = [[0.0 if i == -1.0 else self.margin for i in y]
                 for y in y_list]

        distr_mat = stats.norm(loc=self.output_mat['mu'] * np.vstack(y_hat),
                               scale=self.output_mat['sigma'])
        cutoff_mat = np.array([[0.0 if i == 1.0 else -self.margin for i in y]
                               for y in y_list])
        y_logl = np.sum(distr_mat.sf(cutoff_mat))

        return y_logl

    def get_params(self, deep=True):
        return {'sigma_h': self.sigma_h,
                'prec_distr': self.prec_distr,
                'latent_features': self.R,
                'margin': self.margin}

