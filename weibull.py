import numpy as np


def weibull(t: np.array, ul: float, alpha: float, beta: float) -> np.array:
    return ul * (1 - np.exp(-(t/alpha)**beta))

def d_alpha_weibull(t: np.array, ul: float, alpha: float, beta: float) -> np.array:
    w = weibull(t, ul, alpha, beta)
    return (beta / alpha) * (1 + w) * ((t / alpha)**beta)

def d_beta_weibull(t: np.array, ul: float, alpha: float, beta: float) -> np.array:
    w = weibull(t, ul, alpha, beta)
    return - (1 + w) * ((t / alpha)**beta) * np.log(t / alpha)