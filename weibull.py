import numpy as np


def weibull(t: np.array, ul: float, alpha: float, beta: float) -> np.array:
    return ul * (1 - np.exp(-(t/alpha)**beta))

def d_alpha_weibull(t: np.array, ul: float, alpha: float, beta: float) -> np.array:
    return - (beta / alpha) * np.exp(-(t / alpha)**beta) * ((t / alpha)**beta)

def d_beta_weibull(t: np.array, ul: float, alpha: float, beta: float) -> np.array:
    d_beta = np.exp(-(t / alpha)**beta) * ((t / alpha)**beta) * np.log(t / alpha)
    return np.nan_to_num(d_beta)