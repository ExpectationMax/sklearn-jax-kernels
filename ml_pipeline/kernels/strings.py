"""Implementation of string kernels."""
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
from sklearn.gaussian_process.kernels import GenericKernelMixin



class KMerMatchingKernel(GenericKernelMixin, Kernel):
    def __init__(self, distance_kernel):
        self.distance_kernel = distance_kernel


