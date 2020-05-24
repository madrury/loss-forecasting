import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf


import weibull as wb


class LossDevelopmentCurveModel:

    def __init__(self,
                 initial_ul=90.0,
                 initial_alpha=20.0,
                 initial_beta=1.0,
                 alpha_prior_mean=30.0,
                 beta_prior_mean=1.5,
                 shrinkage=0.0):
        self.parameters = np.array([initial_ul, initial_alpha, initial_beta])
        self.prior_means = np.array([alpha_prior_mean, beta_prior_mean])
        self.shrinkage = shrinkage
        self.log_likelihoods = []

    def fit(self, t: np.array, y: np.array):
        n = len(y)

        ts = np.vstack([t[0:-1], t[1:]]).T
        y_increments = np.diff(np.vstack([y[0:-1], y[1:]]).T, axis=1)

        for _ in range(1000):
            current_forecast_increments = np.diff(wb.weibull(ts, *self.parameters[1:]), axis=1)

            current_dalpha_increments = np.diff(wb.d_alpha_weibull(ts, *self.parameters[1:]), axis=1)
            current_dbeta_increments = np.diff(wb.d_beta_weibull(ts, *self.parameters[1:]), axis=1)
            # current_d2alpha_increments = np.diff(wb.d2_alpha_weibull(ts, *self.parameters[1:]), axis=1)
            # current_d2beta_increments = np.diff(wb.d2_beta_weibull(ts, *self.parameters[1:]), axis=1)
            # current_dalpha_dbeta_increments = np.diff(
            #     wb.d_alpha_d_beta_weibull(ts, *self.parameters[1:]), axis=1
            # )

            gradient = np.array([
                self.dl_dalpha(n, y_increments, current_forecast_increments, current_dalpha_increments),
                self.dl_dbeta(n, y_increments, current_forecast_increments, current_dbeta_increments)
            ])
            penalty = self.shrinkage * (self.parameters[1:] - self.prior_means)
            gradient += penalty

            # current_d2l_d2alpha = self.d2l_d2alpha(
            #     n, y_increments, current_forecast_increments,
            #     current_dalpha_increments, current_d2alpha_increments
            # )
            # current_d2l_d2beta = self.d2l_d2beta(
            #     n, y_increments, current_forecast_increments,
            #     current_dbeta_increments, current_d2beta_increments
            # )
            # current_d2l_dalpha_dbeta = self.d2l_dalpha_dbeta(
            #     n, y_increments, current_forecast_increments,
            #     current_dalpha_increments, current_dbeta_increments, current_dalpha_dbeta_increments
            # )
            # hessian = np.array([
            #     [current_d2l_d2alpha - self.shrinkage, current_d2l_dalpha_dbeta],
            #     [current_d2l_dalpha_dbeta, current_d2l_d2beta - self.shrinkage]
            # ])

            self.parameters[0] = np.sum(y_increments) / np.sum(current_forecast_increments)
            self.parameters[1:] += gradient
            # self.parameters[1:] -= 0.1 * np.linalg.solve(hessian, gradient)

            self.log_likelihoods.append(
                self.likelihood(n, y_increments, current_forecast_increments)
            )

    def likelihood(self, n, y_increments, forecast_increments):
        lhds = (
            y_increments * np.log(self.parameters[0] * forecast_increments)
            - self.parameters[0] * forecast_increments
        )
        return (1/n) * np.sum(lhds)

    # First derivatives of the log-likelihood.
    def dl_du(self, n, y_increments, forecast_increments):
        return (1/n) * (y_increments / self.parameters[0] - forecast_increments)

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
        return - (1/n) * np.sum(y_increments) /self.parameters[0]**2

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