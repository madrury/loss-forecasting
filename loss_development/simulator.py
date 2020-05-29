import numpy as np
import pandas as pd

from loss_development.weibull import weibull


class FraudForcastingDataSimulator:

    def __init__(
        self,
        n_obs_min: int = 5,
        n_obs_max: int = 100,
        ultimate_loss_mean: float = 100.0,
        ultimate_loss_std: float = 10.0,
        alpha_mean: float = 30.0,
        alpha_std: float = 5.0,
        beta_mean: float = 1.5,
        beta_std: float = 0.5
    ):
        self.n_obs_min = n_obs_min
        self.n_obs_max = n_obs_max
        self.ultimate_loss_mean = ultimate_loss_mean
        self.ultimate_loss_std = ultimate_loss_std
        self.alpha_mean = alpha_mean
        self.alpha_std = alpha_std
        self.beta_mean = beta_mean
        self.beta_std = beta_std

    def simulate_multiple_groups(self, n_groups: int) -> pd.DataFrame:
        groups = []
        for group_id in range(n_groups):
            group = self.simulate_one_random_group()
            group['id'] = group_id
            groups.append(group)
        return pd.concat(groups).reset_index(drop=True)

    def simulate_one_random_group(self) -> pd.DataFrame:
        n_obs = np.random.randint(self.n_obs_min, self.n_obs_max)
        ul = np.around(np.random.normal(loc=self.ultimate_loss_mean, scale=self.ultimate_loss_std))
        alpha = np.random.gamma(
            shape=self.alpha_mean**2 / self.alpha_std**2,
            scale=self.alpha_std**2 / self.alpha_mean
        )
        beta = np.random.gamma(
            shape=self.beta_mean**2 / self.beta_std**2,
            scale=self.beta_std**2 / self.beta_mean
        )
        group = self.simulate_one_fixed_group(self.n_obs_max, ul, alpha, beta)
        return pd.DataFrame({
            'expected_ultimate_loss': ul,
            'ultimate_loss': np.max(group),
            'alpha': alpha,
            'beta': beta,
            't': np.arange(0, n_obs),
            'y': group[:n_obs]
        })

    def simulate_one_fixed_group(
        self,
        n_obs: int,
        ultimate_loss: float,
        alpha: float,
        beta: float
    ) -> np.array:
        gs = ultimate_loss * weibull(np.arange(n_obs), alpha, beta)
        dgs = np.zeros(n_obs)
        dgs[1:] = np.diff(gs)
        dy = np.random.poisson(dgs)
        y = np.cumsum(dy)
        return y