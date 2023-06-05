import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
import math
from Kernels import SquaredExponentialKernel,PeriodicKernel, Kernel, AdditiveKernel, MultiplicativeKernel
# train split
from sklearn.model_selection import train_test_split

def get_mv_normal(mean, cov):
    """ Return a multivariate normal distribution with the given mean and covariance matrix """
    return stats.multivariate_normal(mean, cov, allow_singular=True)

def get_random_kernel(n_kernels = 4):
    """ Return a random additive or multiplicative kernel """
    kernel_type = np.random.choice(["additive", "multiplicative"])
    kernels = []
    for i in range(n_kernels):
        component_kernel = np.random.choice([SquaredExponentialKernel, PeriodicKernel, Kernel])
        print(component_kernel)
        if component_kernel == SquaredExponentialKernel:
            n_params = 2
        elif component_kernel == PeriodicKernel:
            n_params = 3
        elif component_kernel == Kernel:
            n_params = 1
        params = np.random.uniform(0, 5, n_params)
        print(f"Creating kernel {component_kernel} with params ({n_params}) {params}")
        kernel = component_kernel(*params)
        kernels.append(kernel)
    if kernel_type == "additive":
        return AdditiveKernel(kernels)
    elif kernel_type == "multiplicative":
        return MultiplicativeKernel(kernels)



def create_random_mv_normal(xs,noise=0.1, kernel = None):
    """ Return a random sample from a multivariate normal distribution
    """
    n_samples = len(xs)
    # Each point in xs is a sample from a 1D normal distribution.
    means = np.zeros(n_samples)
    # Create the covariance matrix
    if kernel is None:
        kernel = get_random_kernel()
    cov = kernel(xs.reshape(-1,1), xs.reshape(-1,1)) + noise * np.eye(n_samples)
    # Create the multivariate normal distribution
    mv_normal = get_mv_normal(means, cov)
    return mv_normal


if __name__ == "__main__":
    seed = None
    as_time_series = False
    noise = 0.2
    nsamples = 200
    interval = (-2,2)
    diff_kernel = None#SquaredExponentialKernel(1,1)
    #diff_kernel.visualize(show=False)

    if seed is not None:
        np.random.seed(seed)
    X = np.random.uniform(interval[0], interval[1], nsamples)
    #X = np.random.normal(0,1, nsamples)
    X = np.sort(X)
    kernel = get_random_kernel()
    kernel.visualize(show=False)
    rvs = create_random_mv_normal(X, noise=0.01, kernel=kernel)
    Y = rvs.rvs(1)
    if not as_time_series:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, shuffle=True)
    else:
        test_sz = 0.2 * len(X)
        X_train = X[:-int(test_sz)]
        X_test = X[-int(test_sz):]
        Y_train = Y[:-int(test_sz)]
        Y_test = Y[-int(test_sz):]
    
    print(f"X_train min: {np.min(X_train)}, max: {np.max(X_train)}")
    if isinstance(diff_kernel, Kernel):
        kernel = diff_kernel
    try:
        cov = kernel(X_train.reshape(-1,1), X_train.reshape(-1,1)) + noise * np.eye(len(X_train))
    except np.linalg.LinAlgError:
        print("Covariance matrix is not positive definite!")
        exit()
    cov_inv = np.linalg.inv(cov)

    prior_distribution = stats.multivariate_normal(np.zeros(len(X_train)), cov, allow_singular=True)
    # Out goal is to refine the prior distribution to a posterior distribution, where the mean is as close as possible to the actual function values
    cov_test_train = kernel(X_test.reshape(-1,1), X_train.reshape(-1,1), check_positive_definite=False)
    cov_train_test = kernel(X_train.reshape(-1,1), X_test.reshape(-1,1), check_positive_definite=False)
    cov_test_test = kernel(X_test.reshape(-1,1), X_test.reshape(-1,1), check_positive_definite=False)

    post_mean = np.zeros(len(X_test)) + cov_test_train @ cov_inv @ Y_train
    post_cov = cov_test_test - cov_test_train @ cov_inv @ cov_train_test + noise * np.eye(len(X_test))

    try:
        kernel.ensure_positive_definite(post_cov)
    except np.linalg.LinAlgError:
        print("Posterior covariance is not positive definite!")
        exit()

    posterior = stats.multivariate_normal(post_mean, post_cov, allow_singular=True)
    predictions = posterior.mean
    cov_diags = np.diag(post_cov)

    # If we are not using time series data, we need to sort the data points by their x values, since during splitting the order of the data points is lost
    if not as_time_series:
        sort_ind = np.argsort(X_test)
        X_test = X_test[sort_ind]
        Y_test = Y_test[sort_ind]
        predictions = predictions[sort_ind]
        cov_diags = cov_diags[sort_ind]


    mae = np.mean(np.abs(predictions - Y_test))
    print(f"MAE: {mae}")

    fig, axes = plt.subplots(1,3)
    # Plot the full data
    axes[0].scatter(X_train, Y_train, label="Training data")
    axes[0].scatter(X_test, Y_test, label="Test data")
    axes[0].legend()
    axes[0].set_title("Full data")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Y value")
    axes[0].grid(True)

    # Plot the distribution of Y values
    axes[1].hist(Y, bins=20)
    axes[1].set_title("Distribution of Y values")
    axes[1].set_xlabel("Y value")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True)
    
    # If we are using a timeseries, plot the last 100 values of train data, followed by predictions and test data
    if as_time_series:
        axes[2].plot(X_train[-100:], Y_train[-100:], label="Training data")
        axes[2].plot(X_test, predictions, label="Predictions")
        axes[2].plot(X_test, Y_test, label="Test data")
        axes[2].legend()
        axes[2].set_title("Time series predictions vs actual values")
        axes[2].set_xlabel("Index") 
        axes[2].set_ylabel("Y value")
    # If we are not using a timeseries, plot the predictions and test data with confidence intervals
    else:
        axes[2].plot(X_test, predictions, label="Predictions")
        axes[2].scatter(X_test, Y_test, label="Test data", color="orange")
        axes[2].fill_between(X_test, predictions - 2 * np.sqrt(cov_diags), predictions + 2 * np.sqrt(cov_diags), alpha=0.5, label="95% confidence interval", color="green")
        axes[2].legend()
        axes[2].set_title("Predictions vs actual values")
        axes[2].set_xlabel("Index")
        axes[2].set_ylabel("Y value")
    axes[2].grid(True)

    plt.show()
