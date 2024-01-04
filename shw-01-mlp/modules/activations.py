import numpy as np
from .base import Module
import scipy.special


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(0, input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        out = np.zeros(input.shape)
        out[input > 0] = 1
        return grad_output * out


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return scipy.special.expit(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * (scipy.special.expit(input))**2 * np.exp(-input)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return scipy.special.softmax(input - np.max(input, axis = 1)[:, None], axis=1)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # https://themaverickmeerkat.com/2019-10-23-Softmax/
        norm_input = input - np.max(input, axis = 1)[:, None]
        p = scipy.special.softmax(norm_input, axis = 1)
        tensor1 = np.einsum('ij,ik->ijk', p, p)
        n = input.shape[1]
        tensor2 = np.einsum('ij,jk->ijk', p, np.eye(n, n))
        dSoftmax = tensor2 - tensor1
        dz = np.einsum('ijk,ik->ij', dSoftmax, grad_output)
        return dz


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        norm_inp = input - np.max(input, axis = 1)[:, None]
        return scipy.special.log_softmax(norm_inp, axis=1)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        norm_input = input - np.max(input, axis = 1)[:, None]
        p = scipy.special.softmax(norm_input, axis=1)
        jacobian = np.einsum('ij,ik->ijk', np.ones(p.shape), p)
        return np.einsum("ij,ijk->ik", grad_output, np.eye(p.shape[1]) - jacobian)
