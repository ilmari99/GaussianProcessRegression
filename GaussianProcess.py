import numpy as np
from Kernels import SquaredExponentialKernel, MaternKernel, PeriodicKernel, Kernel
import scipy.stats as stats
import math

KERNEL_MAP = {
    "SquaredExponential" : SquaredExponentialKernel,
    "Matern" : MaternKernel,
    "Periodic" : PeriodicKernel,
    "Linear" : Kernel,
}

class GaussianProcessRegression:
    """ This class defines a gaussian process regression model.
    """
    def __init__(self, kernel : str | Kernel, sigma_n : float, verbose : bool = False):
        if isinstance(kernel, str):
            # If using the kernel through string id, then default parameters are used
            kernel = KERNEL_MAP[kernel]
        self.kernel = kernel
        self.sigma_n = sigma_n
        self.verbose = verbose
    
    def fit(self, X : np.ndarray, y : np.ndarray, allow_singular = True) -> stats.multivariate_normal:
        """ Fits the GP model to the data, and returns the prior distribution.
        """
        self.X_train = X
        self.y_train = y
        self.K_train = self.kernel(X, X)
        # Add nugget
        self.K_train += self.sigma_n * np.eye(self.K_train.shape[0])
        self.kernel.ensure_positive_definite(self.K_train)
        self.K_train_inv = np.linalg.inv(self.K_train)
        self.prior_mean = np.zeros(len(X))
        self.prior = stats.multivariate_normal(self.prior_mean, self.K_train, allow_singular=allow_singular)
        return self.prior


    def predict(self, X : np.ndarray, allow_singular = True) -> stats.multivariate_normal:
        """ Predicts the output for the given inputs.
        """
        if self.verbose:
            print(f"Predicting X shape {X.shape}")
        K_test_train = self.kernel(X, self.X_train,check_positive_definite=False)
        K_test_test = self.kernel(X, X) + self.sigma_n * np.eye(X.shape[0])
        post_mean = K_test_train @ self.K_train_inv @ self.y_train
        post_K = K_test_test - K_test_train @ self.K_train_inv @ K_test_train.T
        self.kernel.ensure_positive_definite(post_K)
        post = stats.multivariate_normal(post_mean.ravel(), post_K, allow_singular=allow_singular)
        return post
    