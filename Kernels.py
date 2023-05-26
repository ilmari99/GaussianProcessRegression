import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import kv as modified_bessel_first_kind
from scipy.special import gamma as gamma_function

class Kernel:
    """ This class is the base class for all kernels.
    This defines a linear kernel with no hyperparameters.
    """
    def __init__(self, *args):
        self.verbose = True
        pass

    def kernel_function(self, x1, x2):
        return x1 @ x2.T

    def __call__(self, x1, x2, check_positive_definite=True):
        K = self.kernel_function(x1, x2)
        if check_positive_definite:
            self.ensure_positive_definite(K)
        return K

    def __repr__(self):
        return "Kernel"
    
    def ensure_positive_definite(self, x, tol=10**(-8)):
        """ This method checks if the matrix x is positive definite. If not, an error is raised.
        """
        if self.verbose:
            print(f"Checking whether matrix shape {x.shape} is positive definite")
        # If the smallest eigenvalue is smaller than -10^(-6), raise an error
        if np.linalg.eigvalsh(x).min() < -tol:
            print(np.linalg.eigvalsh(x).min())
            raise np.linalg.LinAlgError("Matrix is not positive definite!")
        else:
            return x
        
class SquaredExponentialKernel(Kernel):
    """ This class defines the squared exponential kernel.
    """
    def __init__(self, sigma, l):
        super().__init__()
        self.sigma = sigma
        self.l = l
    
    def kernel_function(self, x1, x2):
        """ Calculates the squared exponential kernel between x1 (n1 x d) and x2 (n2 x d)
        """
        sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        out = self.sigma**2 * np.exp(-0.5/self.l * sqdist)
        return out

class PeriodicKernel(Kernel):
    """ This class defines the periodic kernel.
    """
    def __init__(self, sigma, l, p):
        super().__init__()
        self.sigma = sigma
        self.l = l
        self.p = p
    
    def kernel_function(self, x1, x2):
        """ Calculates the periodic kernel between x1 (n1 x d) and x2 (n2 x d)
        """
        # Compute the pairwise distances between x1 and x2
        dist = cdist(x1 / self.l, x2 / self.l, metric='euclidean')
        
        # Compute the kernel values
        K = self.sigma**2 * np.exp(-2 * np.sin(np.pi * dist / self.p)**2)
        
        print(f"Cov shape: {K.shape}")
        return K
    
class MaternKernel(Kernel):
    """ This class defines the Matern kernel.
    """
    def __init__(self, sigma, l, nu):
        super().__init__()
        self.sigma = sigma
        self.l = l
        self.nu = nu
    
    def kernel_function(self, x1, x2):
        """ Calculates the Matern kernel between x1 (n1 x d) and x2 (n2 x d)
        """
        # Compute the pairwise distances between x1 and x2
        dist = cdist(x1 / self.l, x2 / self.l, metric='euclidean')
        
        # Compute the kernel values
        K = (2**(1-self.nu) / gamma_function(self.nu)) * (np.sqrt(2*self.nu) * dist)**self.nu
        K = self.sigma**2 * modified_bessel_first_kind(self.nu, np.sqrt(2*self.nu) * dist) * K
        
        print(f"Cov shape: {K.shape}")
        return K
