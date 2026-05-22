import numpy as np


# -------------------------------------------------
# Question 1: Joint Gaussian PDF and Marginals
# -------------------------------------------------

def joint_gaussian_pdf(x, y, mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6):
    coeff = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))
    q = (
        ((x - mu_x)**2) / sigma_x**2
        - 2 * rho * ((x - mu_x) * (y - mu_y)) / (sigma_x * sigma_y)
        + ((y - mu_y)**2) / sigma_y**2
    )
    return coeff * np.exp(-q / (2 * (1 - rho**2)))


def marginal_pdf_x(x, mu_x=1, sigma_x=2):
    return (1 / (sigma_x * np.sqrt(2 * np.pi))) * np.exp(-((x - mu_x)**2) / (2 * sigma_x**2))


def marginal_pdf_y(y, mu_y=-2, sigma_y=3):
    return (1 / (sigma_y * np.sqrt(2 * np.pi))) * np.exp(-((y - mu_y)**2) / (2 * sigma_y**2))


def covariance_matrix(sigma_x=2, sigma_y=3, rho=0.6):
    return np.array([
        [sigma_x**2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2]
    ])


def joint_pdf_grid_integral(mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6, n=250):
    x_min, x_max = mu_x - 4 * sigma_x, mu_x + 4 * sigma_x
    y_min, y_max = mu_y - 4 * sigma_y, mu_y + 4 * sigma_y
    x_vals = np.linspace(x_min, x_max, n)
    y_vals = np.linspace(y_min, y_max, n)
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = joint_gaussian_pdf(X, Y, mu_x, mu_y, sigma_x, sigma_y, rho)
    return np.sum(Z) * dx * dy


# -------------------------------------------------
# Question 2: Simulation and Independence
# -------------------------------------------------

def generate_joint_gaussian_samples(
    n=100000,
    mu_x=1,
    mu_y=-2,
    sigma_x=2,
    sigma_y=3,
    rho=0.6,
    seed=0
):
    rng = np.random.default_rng(seed)
    mean = np.array([mu_x, mu_y])
    cov = covariance_matrix(sigma_x, sigma_y, rho)
    samples = rng.multivariate_normal(mean, cov, n)
    return samples[:, 0], samples[:, 1]


def sample_means(x_samples, y_samples):
    return np.mean(x_samples), np.mean(y_samples)


def sample_covariance_matrix(x_samples, y_samples):
    data = np.vstack([x_samples, y_samples])
    return np.cov(data)


def sample_correlation(x_samples, y_samples):
    return np.corrcoef(x_samples, y_samples)[0, 1]


def gaussian_independence_check(rho):
    return rho == 0


def zero_rho_covariance_check(n=100000):
    x, y = generate_joint_gaussian_samples(n=n, rho=0, seed=0)
    cm = sample_covariance_matrix(x, y)
    return bool(abs(cm[0, 1]) < 0.1)


def nonzero_rho_covariance_check(n=100000):
    x, y = generate_joint_gaussian_samples(n=n, rho=0.6, seed=0)
    cm = sample_covariance_matrix(x, y)
    return bool(abs(cm[0, 1] - 0.6 * 2 * 3) < 0.2)
