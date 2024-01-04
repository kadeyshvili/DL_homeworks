import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        return 1 / (input.shape[0] * input.shape[1]) * np.sum((input - target) ** 2)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        return 2 / (input.shape[0] * input.shape[1]) * (input - target)


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """

    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        probabilities = self.log_softmax.compute_output(input)
        q = np.zeros(input.shape)
        q[np.arange(target.shape[0]), target] = 1
        cross_entropy = - np.sum(q * probabilities) / input.shape[0]
        return cross_entropy

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        q = np.zeros(input.shape)
        q[np.arange(target.shape[0]), target] = 1
        return -1 / input.shape[0] * self.log_softmax.compute_grad_input(input, q)
