import numpy as np
import pandas as pd

from weibull import weibull


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
        ul = np.random.normal(loc=self.ultimate_loss_mean, scale=self.ultimate_loss_std)
        alpha = np.random.gamma(
            shape=self.alpha_mean**2 / self.alpha_std**2,
            scale=self.alpha_std**2 / self.alpha_mean
        )
        beta = np.random.gamma(
            shape=self.beta_mean**2 / self.beta_std**2,
            scale=self.beta_std**2 / self.beta_mean
        )
        return pd.DataFrame({
            'ultimate_loss': ul,
            'alpha': alpha,
            'beta': beta,
            't': np.arange(1, n_obs + 2),
            'y': self.simulate_one_fixed_group(n_obs, ul, alpha, beta)
        })

    def simulate_one_fixed_group(
        self,
        n_obs: int,
        ultimate_loss: float,
        alpha: float,
        beta: float
    ) -> np.array:
        dts = np.random.uniform(0.0, 1.0, size=n_obs)
        ts = np.zeros(shape=n_obs + 1)
        ts[1:] = n_obs * np.cumsum(dts) / np.sum(dts)
        ys = weibull(ts, ultimate_loss, alpha, beta)
        return ys