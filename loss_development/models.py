from typing import List

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

import loss_development.weibull as wb
from loss_development.utils import (
    diff_within_groups,
    count_elements_in_groups,
    count_elements_in_group_differences
)


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
                 initial_ultimate_loss=150.0,
                 initial_alpha=20.0,
                 initial_beta=1.0,
                 alpha_prior_mean=30.0,
                 alpha_prior_std=30.0,
                 beta_prior_mean=1.5,
                 beta_prior_std=5.0,
                 shrinkage=0.01,
                 max_iter=10000,
                 rtol=1.0e-6):
        self.learning_rate = learning_rate
        self.parameters = np.array([initial_alpha, initial_beta])
        self.ultimate_losses: np.array
        self.prior_means = np.array([alpha_prior_mean, beta_prior_mean])
        self.shrinkage = shrinkage * np.array([2/alpha_prior_mean, 2/beta_prior_std])
        self.max_iter = max_iter
        self.rtol = rtol
        self.log_likelihoods: List[float] = []
        self.losses: List[float] = []

    def fit(self, t: np.array, y: np.array, groups: np.array):
        self.n = len(y)
        self.unique_groups = np.unique(groups)
        self.group_counts = count_elements_in_groups(groups)
        self.group_difference_counts = count_elements_in_group_differences(self.group_counts)

        # We keep both the individual ultimate losses in a small array, and
        # explode the ultmimate losses into an array with the same shape as the
        # data. Both are useful in different circumsances.
        self.ultimate_losses = np.full(np.unique(groups).shape, 150.0)
        self.ultimate_losses_exploded = np.repeat(self.ultimate_losses, self.group_counts - 1)
        y_increments = diff_within_groups(y, groups)

        for i in range(self.max_iter):
            current_forecast_increments = diff_within_groups(wb.weibull(t, *self.parameters), groups)
            current_dalpha_increments = diff_within_groups(wb.d_alpha_weibull(t, *self.parameters), groups)
            current_dbeta_increments = diff_within_groups(wb.d_beta_weibull(t, *self.parameters), groups)

            gradient = np.array([
                self.dl_dalpha(y_increments, current_forecast_increments, current_dalpha_increments),
                self.dl_dbeta(y_increments, current_forecast_increments, current_dbeta_increments)
            ])

            penalty = self.shrinkage * (self.parameters - self.prior_means)
            self.update_ultimate_losses(y_increments, current_forecast_increments)
            self.ultimate_losses_exploded = np.repeat(self.ultimate_losses, self.group_counts - 1)
            # self.parameters[0] = np.min([np.max([self.parameters[0], 100.0]), 50000.0])
            self.parameters += self.learning_rate * (gradient - penalty)

            self.log_likelihoods.append(
                self.likelihood(y_increments, current_forecast_increments)
            )
            self.losses.append(
                self.loss(y_increments, current_forecast_increments)
            )

            if i > 2 and np.abs(self.log_likelihoods[-1] / self.log_likelihoods[-2] - 1) < self.rtol:
                break

        # final_ultimate_forecast_increments = np.diff(self.forecast(t))
        # self.overdispersion = (1 / self.n) * np.sum(
        #     (final_ultimate_forecast_increments - y_increments)**2 / (final_ultimate_forecast_increments)
        # )
        # self.final_hessian = self.hessian(t, y_increments)

    def update_ultimate_losses(self, y_increments, forecast_increments):
        for idx, group in enumerate(self.unique_groups):
            y_increments_group = y_increments[
                self.group_difference_counts[idx]:self.group_difference_counts[idx+1]
            ]
            forecast_increments_group = forecast_increments[
                self.group_difference_counts[idx]:self.group_difference_counts[idx+1]
            ]
            self.ultimate_losses[idx] += self.learning_rate * (
                # This term is the zero of the ultimate loss component of the
                # gradient. In this dimension, we do not use straightforward
                # gradient descent, instead we immediately update to the zero
                # of the gradient component.
                np.sum(y_increments_group) / np.sum(forecast_increments_group) - self.ultimate_losses[idx]
            )

    def likelihood(self, y_increments, forecast_increments):
        lhds = (
            y_increments * np.log(self.ultimate_losses_exploded * forecast_increments)
            - self.ultimate_losses_exploded * forecast_increments
        )
        return (1 / self.n) * np.sum(lhds)

    def loss(self, y_increments, forecast_increments):
        neg_log_lik = - self.likelihood(y_increments, forecast_increments)
        penalty = 0.5 * np.sum(self.shrinkage * (self.parameters - self.prior_means)**2)
        return neg_log_lik + (1 / self.n) * penalty

    def forecast(self, t, group):
        idx = np.argmax(self.unique_groups == group)
        return self.ultimate_losses[idx] * wb.weibull(t, *self.parameters)

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
    def dl_du(self, y_increments, forecast_increments):
        return (1/self.n) * np.sum(y_increments / self.parameters[0] - forecast_increments)

    def dl_dalpha(self, y_increments, forecast_increments, dalpha_increments):
        dalphas = (
            (y_increments * dalpha_increments) / forecast_increments
            - self.parameters[0] * dalpha_increments
        )
        return (1/self.n) * np.sum(dalphas)

    def dl_dbeta(self, y_increments, forecast_increments, dbeta_increments):
        dbetas = (
            (y_increments * dbeta_increments) / forecast_increments
            - self.parameters[0] * dbeta_increments
        )
        return (1/self.n) * np.sum(dbetas)

    # Second derivatives of the log-likelihood.
    def d2l_d2u(self, y_increments):
        return - (1/self.n) * np.sum(y_increments) / self.parameters[0]**2

    def d2l_du_dalpha(self, dalpha_increments):
        return - (1/self.n) * np.sum(dalpha_increments)

    def d2l_du_dbeta(self, dbeta_increments):
        return - (1/self.n) * np.sum(dbeta_increments)

    def d2l_d2alpha(self, y_increments, forecast_increments, dalpha_increments, d2alpha_increments):
        d2alpha = (
            y_increments * (forecast_increments * d2alpha_increments - dalpha_increments**2)
                / forecast_increments**2
            - self.parameters[0] * d2alpha_increments
        )
        return (1/self.n) * np.sum(d2alpha)

    def d2l_d2beta(self, y_increments, forecast_increments, dbeta_increments, d2beta_increments):
        d2beta = (
            y_increments * (forecast_increments * d2beta_increments - dbeta_increments**2)
                / forecast_increments**2
            - self.parameters[0] * d2beta_increments
        )
        return (1/self.n) * np.sum(d2beta)

    def d2l_dalpha_dbeta(self, y_increments, forecast_increments, dalpha_increments, dbeta_increments, dalpha_dbeta_increments):
        dalpha_dbeta = (
            y_increments * (forecast_increments * dalpha_dbeta_increments - dalpha_increments*dbeta_increments)
                / forecast_increments**2
            - self.parameters[0] * dalpha_dbeta_increments
        )
        return (1/self.n) * np.sum(dalpha_dbeta)

    def hessian(self, t, y_increments):
        # Partial derivatives of forecast increments wrt parameters.
        final_forecast_increments = np.diff(wb.weibull(t, *self.parameters[1:]))
        final_dalpha_increments = np.diff(wb.d_alpha_weibull(t, *self.parameters[1:]))
        final_dbeta_increments = np.diff(wb.d_beta_weibull(t, *self.parameters[1:]))
        final_d2alpha_increments = np.diff(wb.d2_alpha_weibull(t, *self.parameters[1:]))
        final_d2beta_increments = np.diff(wb.d2_beta_weibull(t, *self.parameters[1:]))
        final_dalpha_dbeta_increments = np.diff(wb.d_alpha_d_beta_weibull(t, *self.parameters[1:]))
        # Second partial derivatives of likelihood with respect to parameters.
        final_d2l_d2u = self.d2l_d2u(y_increments)
        final_d2l_du_dalpha = self.d2l_du_dalpha(final_dalpha_increments)
        final_d2l_du_dbeta = self.d2l_du_dbeta(final_dbeta_increments)
        final_d2l_d2alpha = self.d2l_d2alpha(
            y_increments, final_forecast_increments,
            final_dalpha_increments, final_d2alpha_increments
        )
        final_d2l_d2beta = self.d2l_d2beta(
            y_increments, final_forecast_increments,
            final_dbeta_increments, final_d2beta_increments
        )
        final_d2l_dalpha_dbeta = self.d2l_dalpha_dbeta(
            y_increments, final_forecast_increments,
            final_dalpha_increments, final_dbeta_increments, final_dalpha_dbeta_increments
        )
        hessian = np.array([
            [final_d2l_d2u, final_d2l_du_dalpha, final_d2l_du_dbeta],
            [final_d2l_du_dalpha, final_d2l_d2alpha, final_d2l_dalpha_dbeta],
            [final_d2l_du_dbeta, final_d2l_dalpha_dbeta, final_d2l_d2beta]
        ])
        return hessian