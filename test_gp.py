import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

from Kernels import SquaredExponentialKernel, MaternKernel, PeriodicKernel, Kernel
from GaussianProcess import GaussianProcessRegression

# Create a sin wave with noise
def create_sin_wave_with_noise(xshape, yshape, noise=0.1):
    # Randomly sample between 0 and 2pi, to get an array of shape xshape
    X = np.zeros(xshape)
    Y = np.zeros(yshape)
    for i in range(xshape[1]):
        X[:,i] = np.random.uniform(0, 2*math.pi, xshape[0])
    Y = np.sin(X) + np.random.normal(0, noise, yshape)
    # Reduce Y to a 1D array by summing over the columns
    Y = np.sum(Y, axis=1)
    return X, Y

def create_3d_data():
    X_train_shape = (1000,2)
    Y_train_shape = (1000,1)
    X_test_shape = (100,2)
    Y_test_shape = (100,1)
    train_noise_level = 0
    test_noise_level = 0
    X_train, Y_train = create_sin_wave_with_noise(X_train_shape, Y_train_shape,train_noise_level)
    X_test, Y_test = create_sin_wave_with_noise(X_test_shape, Y_test_shape,test_noise_level)
    return X_train, Y_train, X_test, Y_test

def create_2d_data():
    X_train_shape = (1000,1)
    Y_train_shape = (1000,1)
    X_test_shape = (100,1)
    Y_test_shape = (100,1)
    train_noise_level = 0.1
    test_noise_level = 0.1
    X_train, Y_train = create_sin_wave_with_noise(X_train_shape, Y_train_shape,train_noise_level)
    X_test, Y_test = create_sin_wave_with_noise(X_test_shape, Y_test_shape,test_noise_level)
    return X_train, Y_train, X_test, Y_test

def run_3d_GP():
    # Create data
    X_train, Y_train, X_test, Y_test = create_3d_data()
    noise_var = 1e-6
    
    # Define kernel and process
    kernel = SquaredExponentialKernel(sigma = 1, l = 1)
    process = GaussianProcessRegression(kernel, noise_var, verbose=True)

    # Show training data in 3d
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, label="Train data")
    ax.legend()

    # Fit model to training data and plot samples from prior
    prior = np.transpose(process.fit(X_train, Y_train).rvs(10))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    print(f"Prior shape: {prior.shape}")
    for prior_col_ind in range(prior.shape[1]):
        p = prior[:,prior_col_ind]
        ax.plot_trisurf(X_train[:,0], X_train[:,1], p, linewidth=0.2, antialiased=True)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, label="Train data")
    ax.legend()


    # Predict on test data and plot test data and samples from posterior
    posterior_distribution = process.predict(X_test)
    posterior_mean = posterior_distribution.mean
    posterior_cov = posterior_distribution.cov
    posterior_samples = np.transpose(posterior_distribution.rvs(10))
    print(f"Posterior mean shape: {posterior_mean.shape}")
    print(f"Posterior cov shape: {posterior_cov.shape}")
    print(f"Posterior samples shape: {posterior_samples.shape}")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for posterior_col_ind in range(posterior_samples.shape[1]):
        p = posterior_samples[:,posterior_col_ind]
        ax.plot_trisurf(X_test[:,0], X_test[:,1], p, linewidth=0.2, antialiased=True)
    ax.scatter(X_test[:,0], X_test[:,1], Y_test, label="Test data")
    ax.legend()

    plt.show()

def run_2d_GP():
    X_train, Y_train, X_test, Y_test = create_2d_data()
    # Sort data
    sort_ind = np.argsort(X_train[:,0])
    X_train = X_train[sort_ind]
    Y_train = Y_train[sort_ind]
    sort_ind = np.argsort(X_test[:,0])
    X_test = X_test[sort_ind]
    Y_test = Y_test[sort_ind]
    noise_var = 1e-6
    kernel = SquaredExponentialKernel(sigma = 1, l = 1)
    process = GaussianProcessRegression(kernel, noise_var, verbose=True)

    # Plot training data
    fig, ax = plt.subplots()
    ax.scatter(X_train, Y_train, label="Train data")
    ax.legend()
    ax.set_title("Training data")

    # Fit model to training data and plot samples from prior
    prior = np.transpose(process.fit(X_train, Y_train).rvs(10))
    fig, ax = plt.subplots()
    ax.plot(X_train, prior)
    ax.set_title("Prior samples")

    # Predict on test data and plot test data and samples from posterior
    posterior_distribution = process.predict(X_test)
    posterior_mean = posterior_distribution.mean
    posterior_cov = posterior_distribution.cov
    posterior_samples = np.transpose(posterior_distribution.rvs(10))
    print(f"Posterior mean shape: {posterior_mean.shape}")
    print(f"Posterior cov shape: {posterior_cov.shape}")
    print(f"Posterior samples shape: {posterior_samples.shape}")
    fig, ax = plt.subplots()
    for posterior_col_ind in range(posterior_samples.shape[1]):
        p = posterior_samples[:,posterior_col_ind]
        ax.plot(X_test, p)
    ax.scatter(X_test, Y_test, label="Test data")
    ax.legend()
    ax.set_title("Posterior samples")

    plt.show()


if __name__ == "__main__":
    #run_2d_GP()
    run_3d_GP()








