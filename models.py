import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

# np.seterr(all='raise')

import weibull as wb

class LossDevelopmentCurveModel:

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
                 shrinkage=0.1,
                 max_iter=5000,
                 rtol=0.000001):
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
        ts = np.vstack([t[0:-1], t[1:]]).T
        y_increments = np.diff(np.vstack([y[0:-1], y[1:]]).T, axis=1)

        for i in range(self.max_iter):
            current_forecast_increments = np.diff(wb.weibull(ts, *self.parameters[1:]), axis=1)
            current_dalpha_increments = np.diff(wb.d_alpha_weibull(ts, *self.parameters[1:]), axis=1)
            current_dbeta_increments = np.diff(wb.d_beta_weibull(ts, *self.parameters[1:]), axis=1)

            gradient = np.array([
                self.dl_du(n, y_increments, current_forecast_increments),
                self.dl_dalpha(n, y_increments, current_forecast_increments, current_dalpha_increments),
                self.dl_dbeta(n, y_increments, current_forecast_increments, current_dbeta_increments)
            ])

            penalty = self.shrinkage * (self.parameters - self.prior_means)
            self.parameters[0] += self.learning_rate * (
                np.sum(y_increments) / np.sum(current_forecast_increments) - self.parameters[0]
            ) - penalty[0]
            self.parameters[0] = np.min([np.max([self.parameters[0], 5.0]), 200.0])
            self.parameters[1:] += self.learning_rate * gradient[1:] - penalty[1:]

            self.log_likelihoods.append(
                self.likelihood(n, y_increments, current_forecast_increments)
            )
            self.losses.append(
                self.loss(n, y_increments, current_forecast_increments)
            )

            if i > 2 and np.abs(self.losses[-1] / self.losses[-2] - 1) < self.rtol:
                break

        final_forecast_increments = np.diff(wb.weibull(ts, *self.parameters[1:]), axis=1)
        self.overdispersion = (1 / (n - 3)) * np.sum(
            (self.parameters[0] * final_forecast_increments - y_increments)**2
                / (self.parameters[0] * final_forecast_increments)
        )


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
        return np.sqrt(self.overdispersion * np.cumsum(forecast_diffs))

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

    def parameter_variances(self):
        pass
        # hessian = self.hessian(n, ts, y_increments)
        # covariance_matrix = - self.overdispersion * np.linalg.inv(n * hessian)

        # t = np.arange(100)
        # ts = np.vstack([t[0:-1], t[1:]]).T

        # final_dul = wb.weibull(t, *self.parameters[1:])
        # final_dalpha = wb.d_alpha_weibull(t, *self.parameters[1:])
        # final_dbeta = wb.d_beta_weibull(t, *self.parameters[1:])
        # dforecast_increments = np.array([
        #     final_dul,
        #     self.parameters[0] * final_dalpha,
        #     self.parameters[0] * final_dbeta
        # ]).squeeze()

        # forecast_increment_variances = np.einsum(
        #     'ij,jk,ki->i', dforecast_increments.T, covariance_matrix, dforecast_increments
        # )

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