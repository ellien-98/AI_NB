import numpy as np

class NaiveBayes():

    def __init__(self, X, y):
        # rows = number of examples
        # cols = number of features
        self.num_examples, self.num_features =  X.shape
        self.num_classes = len(np.unique(y))
        # self.eps = 1e - 6

    def fit(self, X):
        pass

    def predict(self, X):
        pass

    def density_function(self, x, mean, sigma):
        # calculate probability from gaussian density function
        pass

