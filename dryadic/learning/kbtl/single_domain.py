
"""Using transfer learning to predict one or more labels in a single domain.

"""

from .base import BaseBayesianTransfer

import numpy as np
from scipy import stats
from sklearn.exceptions import NotFittedError

from abc import abstractmethod
from functools import reduce
from operator import add


class BaseSingleDomain(BaseBayesianTransfer):
    """Base class for transfer learning classifiers in single domain.

    Args:

    Attributes:

    """

    def __init__(self,
                 path_keys=None, kernel='rbf', latent_features=5,
                 prec_distr=(1.0, 1.0), sigma_h=0.1, margin=1.0,
                 kern_gamma=-1.85, max_iter=200, stop_tol=1.0):
        self.lambda_mat = None
        self.A_mat = None
        self.H_mat = None

        self.weight_prior = None
        self.weight_mat = None
        self.output_mat = None

        self.X = None
        self.kernel_mat = None
        self.kkt_mat = None
        self.kern_size = None
        self.samp_count = None

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

        # makes sure training labels are of the correct format
        if isinstance(y, list) or len(y.shape) == 1:
            if not all(yval in [True, False] for yval in y):
                raise TypeError("Output label values must be all boolean!")

            elif not np.isin(y, [True, False]).all():
                raise TypeError("Output label array must be boolean!")

        self.X = X
        self.expr_genes = expr_genes
        self.path_keys = path_keys

        self.kernel_mat = self.compute_kernels(
            X, expr_genes=expr_genes, path_keys=path_keys)

        self.kkt_mat = self.kernel_mat @ self.kernel_mat.transpose()
        self.kern_size = self.kernel_mat.shape[0]
        self.samp_count = self.kernel_mat.shape[1]

        y_list = np.array(y, dtype=np.float) * 2 - 1
        if len(y_list.shape) == 1:
            y_list = y_list.reshape(-1, 1)

        # initializes matrix of posterior distributions of precision priors
        # for the projection matrix
        self.lambda_mat = {'alpha': np.full((self.kern_size, self.R),
                                            self.prec_distr[0] + 0.5),
                           'beta': np.full((self.kern_size, self.R),
                                           self.prec_distr[1])}

        self.pheno_count = y_list.shape[1]
        self.output_mat = self.init_output_mat(y_list)

        # initializes posteriors of precision priors for coupled
        # classification matrices
        self.weight_prior = {'alpha': np.full((self.R + 1, self.pheno_count),
                                              self.prec_distr[0] + 0.5),
                             'beta': np.full((self.R + 1, self.pheno_count),
                                             self.prec_distr[1])}

        self.weight_mat = {
            'mu': np.vstack((np.zeros((1, self.pheno_count)),
                             np.random.randn(self.R, self.pheno_count))),
            'sigma': np.tile(np.eye(self.R + 1), (self.pheno_count, 1, 1))
            }

        self.A_mat = {'mu': np.random.randn(self.kern_size, self.R),
                      'sigma': np.array(np.eye(self.kern_size)[..., None]
                                        * ([1] * self.R)).transpose()}

        self.H_mat = {'mu': np.random.randn(self.R, self.samp_count),
                      'sigma': np.eye(self.R)}

        # proceeds with inference using variational Bayes for the given
        # number of iterations
        cur_iter = 1
        old_log_like = float('-inf')
        log_like_stop = False
        while cur_iter <= 10 and not log_like_stop:

            new_lambda = self.update_precision_priors(
                self.lambda_mat, self.A_mat,
                self.prec_distr[0], self.prec_distr[1]
                )
            new_proj, new_latent = self.update_latent(new_lambda)

            new_weight_prior = self.update_precision_priors(
                self.weight_prior, self.weight_mat, *self.prec_distr)
            new_weight = self.update_weights(new_weight_prior, new_latent)
            new_output = self.update_output(new_latent, new_weight, y_list)

            self.lambda_mat = new_lambda
            self.A_mat = new_proj
            self.H_mat = new_latent

            self.weight_prior = new_weight_prior
            self.weight_mat = new_weight
            self.output_mat = new_output

            # every ten update iterations, check the likelihood of the model
            # given current latent variable distributions
            if (cur_iter % 2) == 0:
                cur_log_like = self.log_likelihood(y_list)
                print(cur_log_like)

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
        kern_dist = self.compute_kernels(x_mat=self.X, y_mat=X,
                                         expr_genes=self.expr_genes)

        h_mu = self.A_mat['mu'].transpose() @ kern_dist
        f_mu = [np.zeros(X.shape[0]) for _ in range(self.pheno_count)]
        f_sigma = [np.zeros(X.shape[0]) for _ in range(self.pheno_count)]

        for i in range(self.pheno_count):
            f_mu[i] = np.dot(
                np.vstack(([1 for _ in range(X.shape[0])],
                           h_mu)).transpose(),
                self.weight_mat['mu'][:, i]
                )

            f_sigma[i] = 1.0 + np.diag(
                np.dot(
                    np.dot(
                        np.vstack(([1 for _ in range(X.shape[0])],
                                   h_mu)).transpose(),
                        self.weight_mat['sigma'][i, :, :]),
                    np.vstack(([1 for _ in range(X.shape[0])], h_mu))
                    )
                )

        return f_mu, f_sigma

    def predict_proba(self, X):
        f_mu, f_sigma = self.predict_labels(X)
        pred_arr = np.zeros((X.shape[0], self.pheno_count))

        for i in range(self.pheno_count):
            pred_p, pred_n = self.get_pred_class_probs(f_mu[i], f_sigma[i])
            pred_arr[:, i] = pred_p / (pred_p + pred_n)

        return pred_arr

    @abstractmethod
    def init_output_mat(self, y_list):
        """Initialize posterior distributions of the output predictions."""

    def update_latent(self, new_prior):
        new_projection = {'mu': np.zeros(self.A_mat['mu'].shape),
                          'sigma': np.zeros(self.A_mat['sigma'].shape)}
        new_latent = {k: np.zeros(hmat.shape)
                      for k, hmat in self.H_mat.items()}

        prior_expect = new_prior['alpha'] / new_prior['beta']
        prior_expect = prior_expect.transpose().tolist()

        for i in range(self.R):
            new_projection['sigma'][i, :, :] = np.linalg.inv(
                np.diag(prior_expect[i])
                + (self.kkt_mat / self.sigma_h)
                )

            new_projection['mu'][:, i] = np.dot(
                new_projection['sigma'][i, :, :],
                np.dot(self.kernel_mat,
                       self.H_mat['mu'][i, :])
                / self.sigma_h
                )

        new_latent['sigma'] = np.linalg.inv(
            np.diag([self.sigma_h ** -1 for _ in range(self.R)])
            + reduce(add, [np.outer(self.weight_mat['mu'][1:, i],
                                    self.weight_mat['mu'][1:, i])
                           + self.weight_mat['sigma'][i][1:, 1:]
                      for i in range(self.pheno_count)])
            )

        new_latent['mu'] = np.dot(
            new_latent['sigma'],
            np.dot(new_projection['mu'].transpose(),
                   self.kernel_mat) / self.sigma_h
            + reduce(add, [np.outer(self.weight_mat['mu'][1:, i],
                                    self.output_mat['mu'][:, i])
                           - np.repeat(
                               a=np.array([[
                                   x * self.weight_mat['mu'][0, i] + y
                                   for x, y in zip(
                                       self.weight_mat['mu'][1:, i],
                                       self.weight_mat['sigma'][i, 1:, 0]
                                    )]]),
                               repeats=self.samp_count, axis=0
                            ).transpose()
                           for i in range(self.pheno_count)])
            )

        return new_projection, new_latent

    def update_weights(self, new_prior, new_latent):
        new_weight = {'mu': np.zeros(new_prior['alpha'].shape),
                      'sigma': np.zeros((self.pheno_count,
                                         self.R + 1, self.R + 1))}

        h_sum = np.sum(new_latent['mu'], axis=1)
        hht_mat = new_latent['mu'] @ new_latent['mu'].transpose()
        hht_mat += new_latent['sigma'] * self.samp_count

        for i in range(self.pheno_count):
            new_weight['sigma'][i, 0, 0] = (
                new_prior['alpha'][0, i] / new_prior['beta'][0, i]
                + self.samp_count
                )

            new_weight['sigma'][i, 1:, 0] = h_sum
            new_weight['sigma'][i, 0, 1:] = h_sum

            new_weight['sigma'][i, 1:, 1:] = hht_mat
            new_weight['sigma'][i, 1:, 1:] += np.diag(
                new_prior['alpha'][1:, i] / new_prior['beta'][1:, i])

            new_weight['sigma'][i, :, :] = np.linalg.inv(
                new_weight['sigma'][i, :, :])

            new_weight['mu'][:, i] = np.dot(
                new_weight['sigma'][i, :, :],
                np.dot(np.vstack([np.ones(self.samp_count),
                                  new_latent['mu']]),
                       self.output_mat['mu'][:, i])
                )

        return new_weight

    @abstractmethod
    def update_output(self, latent_mat, weight_mat, y_list):
        """Update the predicted output labels."""

    @abstractmethod
    def get_y_logl(self, y_list):
        """Computes the log-likelihood of a set of output labels given
           the current model state.
        """

    def log_likelihood(self, y_list):
        """Computes the log-likelihood of the current model state.

        Args:

        Returns:

        """
        if self.lambda_mat is None:
            raise NotFittedError(
                "Can't compute model likelihood before fitting!")

        # precision prior distribution given precision hyper-parameters
        prec_distr = stats.gamma(a=self.prec_distr[0],
                                 scale=self.prec_distr[1] ** -1.0)

        # likelihood of projection matrix precision priors given
        # precision hyper-parameters
        lambda_logl = np.sum(
            prec_distr.logpdf(self.lambda_mat['alpha']
                              / self.lambda_mat['beta'])
            )

        # likelihood of projection matrix values given their precision priors
        a_logl = np.sum(stats.norm(
            loc=0, scale=self.lambda_mat['beta'] / self.lambda_mat['alpha'])
            .logpdf(self.A_mat['mu']))

        # likelihood of latent feature matrix given kernel matrix,
        # projection matrix, and standard deviation hyper-parameter
        h_logl = np.sum(stats.norm(
            loc=self.A_mat['mu'].transpose() @ self.kernel_mat,
            scale=self.sigma_h ** 0.5
            ).logpdf(self.H_mat['mu']))

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
        f_logl = np.sum(stats.norm(
            loc=(self.weight_mat['mu'][1:, :].transpose()
                 @ self.H_mat['mu']
                 + np.vstack(self.weight_mat['mu'][0, :])).transpose(),
            scale=1
            ).logpdf(self.output_mat['mu']))

        return (lambda_logl + a_logl + h_logl + weight_prior_logl
                + weight_logl + f_logl + self.get_y_logl(y_list))

    def get_output_distr(self, stat_list):
        """Gets the cumulative PDF of the output labels.

        Returns:
            out_probs (list): For each task, a list of (x, prob) pairs
        """
        out_probs = [[np.zeros(500) for _ in stat_list]
                     for _ in range(self.pheno_count)]

        # for each task, get the posterior distribution for each predicted
        # output label
        for i in range(self.pheno_count):
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
            start_indx = self.samp_count * i
            end_indx = self.samp_count * (i + 1)
            proj_list[pk] = proj_max[start_indx:end_indx]

        return proj_list


class SingleDomain(BaseSingleDomain):

    def init_output_mat(self, y_list):
        return {'mu': ((abs(np.random.randn(*y_list.shape)) + self.margin)
                       * np.sign(y_list)),
                'sigma': np.ones(y_list.shape)}

    def update_output(self, new_latent, weight_mat, y_list):
        """Update the predicted output labels."""
        new_output = {k: np.zeros(omat.shape)
                      for k, omat in self.output_mat.items()}

        # gets the margin boundaries for each output label class
        lu_list = {
            'lower': np.vectorize(
                lambda y: -1e40 if y <= 0 else self.margin)(y_list),
            'upper': np.vectorize(
                lambda y: 1e40 if y >= 0 else -self.margin)(y_list)
            }

        for i in range(self.pheno_count):
            f_raw = np.dot(weight_mat['mu'][1:, i], new_latent['mu'])
            f_raw += weight_mat['mu'][0, i]

            alpha_norm = (lu_list['lower'][:, i] - f_raw).tolist()
            beta_norm = (lu_list['upper'][:, i] - f_raw).tolist()
            norm_factor = [stats.norm.cdf(b) - stats.norm.cdf(a)
                           if a != b else 1.0
                           for a, b in zip(alpha_norm, beta_norm)]
            norm_factor = [1.0 if x == 0 else x for x in norm_factor]

            new_output['mu'][:, i] = [
                f + ((stats.norm.pdf(a) - stats.norm.pdf(b)) / n)
                for a, b, n, f in
                zip(alpha_norm, beta_norm, norm_factor, f_raw.tolist())
                ]
            new_output['sigma'][:, i] = [
                1.0 + (a * stats.norm.pdf(a) - b * stats.norm.pdf(b)) / n
                - ((stats.norm.pdf(a) - stats.norm.pdf(b)) ** 2) / n ** 2
                for a, b, n in zip(alpha_norm, beta_norm, norm_factor)
                ]

        return new_output

    def get_y_logl(self, y_list):
        return np.sum(stats.norm(
            loc=self.output_mat['mu'] * np.vstack(y_list),
            scale=self.output_mat['sigma']
            ).logsf(1))


class SingleDomainAsym(BaseSingleDomain):
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
                      / (np.prod(y_sigma[0]) ** (1.0 / self.samp_count))
                      for i in range(self.pheno_count)])
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
                    ), repeats=self.samp_count, axis=0).transpose())
                 / y_sigma[i]
                 for i in range(self.pheno_count)]
                )
            )

        return new_latent

    def update_output(self, latent_mat, weight_mat, y_list):
        """Update the predicted output labels."""
        new_output = {k: np.zeros(self.output_mat[k].shape)
                      for k in self.output_mat}

        for i in range(self.pheno_count):
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

