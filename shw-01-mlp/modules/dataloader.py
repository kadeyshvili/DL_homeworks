from math import ceil
from sklearn.utils import shuffle


class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """

    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return ceil(self.X.shape[0] / self.batch_size)

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return self.X.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        # your code here ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        self.batch_id = 0
        if self.shuffle:
            self.X, self.y = shuffle(self.X, self.y, random_state=0)
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        # your code here ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        if self.X.shape[0] - self.batch_size * self.batch_id > 0:
            left, right = self.batch_size * self.batch_id, self.batch_size * (self.batch_id + 1)
            self.batch_id += 1
            return self.X[left:right], self.y[left:right]
        raise StopIteration
