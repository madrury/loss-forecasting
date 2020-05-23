import numpy as np


def weibull(t: np.array, alpha: float, beta: float) -> np.array:
    return 1 - np.exp(-(t/alpha)**beta)

def d_alpha_weibull(t: np.array, alpha: float, beta: float) -> np.array:
    return - (beta / alpha) * np.exp(-(t / alpha)**beta) * ((t / alpha)**beta)

def d_beta_weibull(t: np.array, alpha: float, beta: float) -> np.array:
    # When t=0 this will be a logarithm of zero, which will throw a divide by
    # zero warning when computing the logarithm. The pardial d/dbeta is zero at
    # t=0, so we silence the warning then infer the zero.
    with np.errstate(divide='ignore'):
        log_factor = np.nan_to_num(np.log(t / alpha))
    return np.exp(-(t / alpha)**beta) * ((t / alpha)**beta) * log_factor

def d2_alpha_weibull(t: np.array, alpha: float, beta: float):
    pass

def d2_beta_weibull(t: np.array, alpha: float, beta: float):
    pass

def d_alpha_d_beta_weibull(t: np.array, alpha: float, beta: float):
    pass