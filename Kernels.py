import numpy as np
import matplotlib.pyplot as plt

class Kernel:
    """ This class is the base class for all kernels.
    This defines a linear kernel with no hyperparameters.
    """
    def __init__(self, *args, **kwargs):
        self.verbose = kwargs.get("verbose", True)
        self.name = kwargs.get("name", "Kernel")

    def kernel_function(self, x1, x2):
        """ Calculates the kernel between x1 (1 x d) and x2 (1 x d) and returns a scalar
        """
        return x1 @ x2.T

    def __call__(self, x1, x2, check_positive_definite=True):
        K = np.zeros((x1.shape[0], x2.shape[0]))
        for i in range(x1.shape[0]):
            for j in range(x2.shape[0]):
                K[i,j] = self.kernel_function(x1[i,:], x2[j,:])
        if check_positive_definite:
            self.ensure_positive_definite(K)
        return K

    def __repr__(self):
        return f"{self.name}()"
    
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
        
    def visualize(self,X1 = None,X2 = None, show = True):
        """ Show a heatmap of the kernel function when applied to two matrices
        """
        if X1 is None:
            X1 = np.arange(-5,5,0.1).reshape(-1,1)
        if X2 is None:
            X2 = np.arange(-5,5,0.1).reshape(-1,1)
        K = self(X1,X2)
        fig,ax = plt.subplots()
        im = ax.imshow(K)
        ax.set_title(f"Heatmap of {self.name}")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        fig.colorbar(im)
        if show:
            plt.show()
        return fig,ax

    
class AdditiveKernel(Kernel):
    def __init__(self, kernels, *args, **kwargs):
        self.kernels = kernels
        if "name" not in kwargs:
            kwargs["name"] = "AdditiveKernel"
        super().__init__(*args, **kwargs)

    def kernel_function(self, x1, x2):
        return sum([kernel.kernel_function(x1, x2) for kernel in self.kernels])
    
class MultiplicativeKernel(Kernel):
    def __init__(self, kernels, *args, **kwargs):
        self.kernels = kernels
        if "name" not in kwargs:
            kwargs["name"] = "MultiplicativeKernel"
        super().__init__(*args, **kwargs)

    def kernel_function(self, x1, x2):
        return np.prod([kernel.kernel_function(x1, x2) for kernel in self.kernels])

        
class SquaredExponentialKernel(Kernel):
    """ This class defines the squared exponential kernel.
    """
    def __init__(self, sigma, l,*args, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "SquaredExponentialKernel"
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        self.l = l
    
    def kernel_function(self, x1, x2):
        """ Calculates the squared exponential kernel between x1 (1 x d) and x2 (1 x d) and returns a scalar
        """
        out = self.sigma**2 * np.exp(-0.5 * np.sum((x1 - x2)**2) / self.l**2)
        return out


class PeriodicKernel(Kernel):
    """ This class defines the periodic kernel.
    """
    def __init__(self, sigma, l, p, *args, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "PeriodicKernel"
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        self.l = l
        self.p = p
    
    def kernel_function(self, x1, x2):
        """ Calculates the periodic kernel between x1 (1 x d) and x2 (1 x d) and returns a scalar
        """
        in_exp = -2 * np.sin(np.linalg.norm(x1 - x2) / (2*self.p))**2 / self.l**2
        out = self.sigma**2 * np.exp(in_exp)
        return out
