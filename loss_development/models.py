import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

import loss_development.weibull as wb


class LossDevelopmentCurveModel:
    """An implementation of the loss forecasting model from this paper:

        https://www.casact.org/pubs/forum/03fforum/03ff041.pdf

    This model takes a sequence of loss values y_i observed over discrete
    time increments t_i and fits a curve of the form:

        f(t) = ultimate_loss * weibull_pdf(t; alpha, beta)

    by maximum likelihood, optimizing the parameters (ultimalte_loss, alpha,
    beta). The probability model underlying the application of maximum
    likelihood is:

        y_{i+1} - y_i ~ Poisson(lambda=(f(t+1) - f(t)))

    The model is fit with a modified version of grandient ascent. Parameters
    alpha and beta are fit iteratively by moving along the gradient, but the
    partial derivative d likelihood / d ultimate_loss has an analytically
    calulcatable zero, so in our descent we solve for the ultimate_loss
    explicitly that zero's out the partial derivative.

    Parameters
    ----------
    learning_rate: float
      Learning rate for the gradient ascent.

    initial_ul, initial_alpha, initial_beta: float
      Initial values for parameters to start the optimization.

    ul_prior_mean, alpha_prior_mean, beta_prior_mean: float
      Parameter values to regularize the model towards during optimization.

    ul_prior_std, alpha_prior_std, beta_prior_std: float
      Inverely related to the strengths of regularization to apply when
      shrinking these parameters towards their prior values.

    shrinkate: floar
      Overall strength of regulariation to apply to all three parameters.

    max_iter: int
      Maximum number of gradient descent iterations.

    trol: float
      Relative tolerance of model convergence criteria.
    """
    def __init__(self,
                 learning_rate=0.05,
                 initial_ul=150.0,
                 initial_alpha=20.0,
                 initial_beta=1.0,
                 ul_prior_mean=150.0,
                 ul_prior_std=40.0,
                 alpha_prior_mean=30.0,
                 alpha_prior_std=30.0,
                 beta_prior_mean=1.5,
                 beta_prior_std=5.0,
                 shrinkage=0.01,
                 max_iter=10000,
                 rtol=1.0e-6):
        self.learning_rate = learning_rate
        self.parameters = np.array([initial_ul, initial_alpha, initial_beta])
        self.prior_means = np.array([ul_prior_mean, alpha_prior_mean, beta_prior_mean])
        self.shrinkage = shrinkage * np.array([2/ul_prior_std, 2/alpha_prior_mean, 2/beta_prior_std])
        self.max_iter = max_iter
        self.rtol = rtol
        self.log_likelihoods = []
        self.losses = []

    def fit(self, t: np.array, y: np.array):
        assert len(t) == len(y)
        n = len(y)
        y_increments = np.diff(y)
        ts = np.vstack([t[0:-1], t[1:]]).T

        for i in range(self.max_iter):
            current_forecast_increments = np.diff(wb.weibull(t, *self.parameters[1:]))
            current_dalpha_increments = np.diff(wb.d_alpha_weibull(t, *self.parameters[1:]))
            current_dbeta_increments = np.diff(wb.d_beta_weibull(t, *self.parameters[1:]))

            gradient = np.array([
                self.dl_du(n, y_increments, current_forecast_increments),
                self.dl_dalpha(n, y_increments, current_forecast_increments, current_dalpha_increments),
                self.dl_dbeta(n, y_increments, current_forecast_increments, current_dbeta_increments)
            ])

            penalty = self.shrinkage * (self.parameters - self.prior_means)
            self.parameters[0] += self.learning_rate * (
                # This term is the zero of the ultimate loss component of the
                # gradient. In this dimension, we do not use straightforward
                # gradient descent, instead we immediately update to the zero
                # of the gradient component.
                np.sum(y_increments) / np.sum(current_forecast_increments) - self.parameters[0]
                 - penalty[0])
            # self.parameters[0] = np.min([np.max([self.parameters[0], 100.0]), 50000.0])
            self.parameters[1:] += self.learning_rate * (gradient[1:] - penalty[1:])

            self.log_likelihoods.append(
                self.likelihood(n, y_increments, current_forecast_increments)
            )
            self.losses.append(
                self.loss(n, y_increments, current_forecast_increments)
            )

            if i > 2 and np.abs(self.log_likelihoods[-1] / self.log_likelihoods[-2] - 1) < self.rtol:
                break

        final_ultimate_forecast_increments = np.diff(self.forecast(t))
        self.overdispersion = (1 / n) * np.sum(
            (final_ultimate_forecast_increments - y_increments)**2 / (final_ultimate_forecast_increments)
        )
        self.final_hessian = self.hessian(n, ts, y_increments)
        self.n = n

    def likelihood(self, n, y_increments, forecast_increments):
        lhds = (
            y_increments * np.log(self.parameters[0] * forecast_increments)
            - self.parameters[0] * forecast_increments
        )
        return (1/n) * np.sum(lhds)

    def loss(self, n, y_increments, forecast_increments):
        neg_log_lik = - self.likelihood(n, y_increments, forecast_increments)
        penalty = 0.5 * np.sum(self.shrinkage * (self.parameters - self.prior_means)**2)
        return neg_log_lik + (1/n) * penalty

    def forecast(self, t):
        return self.parameters[0] * wb.weibull(t, *self.parameters[1:])

    def forecast_interval(self, t):
        forecast_diffs = np.zeros(len(t))
        forecast_diffs[1:] = np.diff(self.forecast(t))
        return (
            np.sqrt(self.overdispersion * np.cumsum(forecast_diffs))
            + np.sqrt(self.forecast_parameter_variances(t))
        )

    def forecast_parameter_variances(self, t):
        covariance_matrix = - self.overdispersion * np.linalg.inv(self.n * self.final_hessian)

        final_dul = wb.weibull(t, *self.parameters[1:])
        final_dalpha = wb.d_alpha_weibull(t, *self.parameters[1:])
        final_dbeta = wb.d_beta_weibull(t, *self.parameters[1:])

        dforecast_increments = np.array([
            final_dul,
            self.parameters[0] * final_dalpha,
            self.parameters[0] * final_dbeta
        ]).squeeze()

        forecast_increment_variances = np.einsum(
            'ij,jk,ki->i', dforecast_increments.T, covariance_matrix, dforecast_increments
        )
        # We sometimes get a non-definate hessian at the final parameter
        # values, which leads to negative variance estimates. We take a
        # conservative approach here, and use the maximal value of the
        # estimated variance to fill in any such issues.
        return np.maximum.accumulate(np.nan_to_num(forecast_increment_variances))

    # First derivatives of the log-likelihood.
    def dl_du(self, n, y_increments, forecast_increments):
        return (1/n) * np.sum(y_increments / self.parameters[0] - forecast_increments)

    def dl_dalpha(self, n, y_increments, forecast_increments, dalpha_increments):
        dalphas = (
            (y_increments * dalpha_increments) / forecast_increments
            - self.parameters[0] * dalpha_increments
        )
        return (1/n) * np.sum(dalphas)

    def dl_dbeta(self, n, y_increments, forecast_increments, dbeta_increments):
        dbetas = (
            (y_increments * dbeta_increments) / forecast_increments
            - self.parameters[0] * dbeta_increments
        )
        return (1/n) * np.sum(dbetas)

    # Second derivatives of the log-likelihood.
    def d2l_d2u(self, n, y_increments):
        return - (1/n) * np.sum(y_increments) / self.parameters[0]**2

    def d2l_du_dalpha(self, n, dalpha_increments):
        return - (1/n) * np.sum(dalpha_increments)

    def d2l_du_dbeta(self, n, dbeta_increments):
        return - (1/n) * np.sum(dbeta_increments)

    def d2l_d2alpha(self, n, y_increments, forecast_increments, dalpha_increments, d2alpha_increments):
        d2alpha = (
            y_increments * (forecast_increments * d2alpha_increments - dalpha_increments**2)
                / forecast_increments**2
            - self.parameters[0] * d2alpha_increments
        )
        return (1/n) * np.sum(d2alpha)

    def d2l_d2beta(self, n, y_increments, forecast_increments, dbeta_increments, d2beta_increments):
        d2beta = (
            y_increments * (forecast_increments * d2beta_increments - dbeta_increments**2)
                / forecast_increments**2
            - self.parameters[0] * d2beta_increments
        )
        return (1/n) * np.sum(d2beta)

    def d2l_dalpha_dbeta(self, n, y_increments, forecast_increments, dalpha_increments, dbeta_increments, dalpha_dbeta_increments):
        dalpha_dbeta = (
            y_increments * (forecast_increments * dalpha_dbeta_increments - dalpha_increments*dbeta_increments)
                / forecast_increments**2
            - self.parameters[0] * dalpha_dbeta_increments
        )
        return (1/n) * np.sum(dalpha_dbeta)

    def hessian(self, n, ts, y_increments):
        # Partial derivatives of forecast increments wrt parameters.
        final_forecast_increments = np.diff(wb.weibull(ts, *self.parameters[1:]), axis=1)
        final_dalpha_increments = np.diff(wb.d_alpha_weibull(ts, *self.parameters[1:]), axis=1)
        final_dbeta_increments = np.diff(wb.d_beta_weibull(ts, *self.parameters[1:]), axis=1)
        final_d2alpha_increments = np.diff(wb.d2_alpha_weibull(ts, *self.parameters[1:]), axis=1)
        final_d2beta_increments = np.diff(wb.d2_beta_weibull(ts, *self.parameters[1:]), axis=1)
        final_dalpha_dbeta_increments = np.diff(
            wb.d_alpha_d_beta_weibull(ts, *self.parameters[1:]), axis=1
        )
        # Second partial derivatives of likelihood with respect to parameters.
        final_d2l_d2u = self.d2l_d2u(n, y_increments)
        final_d2l_du_dalpha = self.d2l_du_dalpha(n, final_dalpha_increments)
        final_d2l_du_dbeta = self.d2l_du_dbeta(n, final_dbeta_increments)
        final_d2l_d2alpha = self.d2l_d2alpha(
            n, y_increments, final_forecast_increments,
            final_dalpha_increments, final_d2alpha_increments
        )
        final_d2l_d2beta = self.d2l_d2beta(
            n, y_increments, final_forecast_increments,
            final_dbeta_increments, final_d2beta_increments
        )
        final_d2l_dalpha_dbeta = self.d2l_dalpha_dbeta(
            n, y_increments, final_forecast_increments,
            final_dalpha_increments, final_dbeta_increments, final_dalpha_dbeta_increments
        )
        hessian = np.array([
            [final_d2l_d2u, final_d2l_du_dalpha, final_d2l_du_dbeta],
            [final_d2l_du_dalpha, final_d2l_d2alpha, final_d2l_dalpha_dbeta],
            [final_d2l_du_dbeta, final_d2l_dalpha_dbeta, final_d2l_d2beta]
        ])
        return hessian