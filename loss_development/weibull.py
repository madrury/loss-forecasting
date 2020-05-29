import numpy as np


def weibull(t: np.array, alpha: float, beta: float) -> np.array:
    return 1 - np.exp(-(t / alpha)**beta)

def d_alpha_weibull(t: np.array, alpha: float, beta: float) -> np.array:
    ratio_power = (t / alpha)**beta
    return - (beta / alpha) * np.exp(- ratio_power) * ratio_power

def d_beta_weibull(t: np.array, alpha: float, beta: float) -> np.array:
    # When t=0 this will be a logarithm of zero, which will throw a divide by
    # zero warning when computing the logarithm. The pardial d/dbeta is zero at
    # t=0, so we silence the warning then infer the zero by setting a logarithm
    # term to a large negative number.
    with np.errstate(divide='ignore'):
        log_factor = np.nan_to_num(np.log(t / alpha))
        log_factor[t == 0] = -100
    ratio_power = (t / alpha)**beta
    return np.exp(- ratio_power) * ratio_power * log_factor

def d2_alpha_weibull(t: np.array, alpha: float, beta: float):
    ratio_power = (t / alpha)**beta
    return - (beta / alpha**2) * ratio_power * np.exp(- ratio_power) * (beta * ratio_power - beta - 1)

def d2_beta_weibull(t: np.array, alpha: float, beta: float):
    with np.errstate(divide='ignore'):
        log_factor = np.nan_to_num(np.log(t / alpha))
        log_factor[t == 0] = -100
    ratio_power = (t / alpha)**beta
    return - ratio_power * np.exp(- ratio_power) * log_factor**2 * (ratio_power - 1)

def d_alpha_d_beta_weibull(t: np.array, alpha: float, beta: float):
    with np.errstate(divide='ignore'):
        log_factor = np.nan_to_num(np.log(t / alpha))
        log_factor[t == 0] = -100
    ratio_power = (t / alpha)**beta
    return (1 / alpha) * ratio_power * np.exp(- ratio_power) * (beta * ratio_power * log_factor - beta * log_factor - 1)