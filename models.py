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
                 shrinkage=0.1):
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

            gradient = np.array([
                self.dl_dalpha(n, y_increments, current_forecast_increments, current_dalpha_increments),
                self.dl_dbeta(n, y_increments, current_forecast_increments, current_dbeta_increments)
            ])
            penalty = self.shrinkage * (self.parameters[1:] - self.prior_means)

            self.parameters[0] = np.sum(y_increments) / np.sum(current_forecast_increments)
            self.parameters[1:] = self.parameters[1:] + 0.1 * gradient - penalty

            self.log_likelihoods.append(
                self.likelihood(n, y_increments, current_forecast_increments)
            )

    def likelihood(self, n, y_increments, forecast_increments):
        lhds = (
            y_increments * np.log(self.parameters[0] * forecast_increments)
            - self.parameters[0] * forecast_increments
        )
        return (1/n) * np.sum(lhds)

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