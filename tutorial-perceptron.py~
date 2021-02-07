import numpy as np

class Perceptron:

    ''' Init func w/ default learning rate and no. of iterations
    '''
    def __init__(self, learning_rate = 0.01, n_iters  = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    ''' Fit func
    '''
    def fit(self, X, y):
        # X is a nd array of size m x n (m - no. of rows(=samples)
        # n - no. of columns(= no. of features))
        n_samples, n_features = X.shape

        # init weights
        # for each feature we put a zero for our weight
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    ''' Prediction function
    '''
    def predict(self, X):
        # we use a linear function (dot product)
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    ''' Step func that can be applied on np arrays
    '''
    def _unit_step_func(slef, x):
        # return 1 if x >= 0, otherwise return zero
        # works for single sample and multiple samples
        # in a vector
        return np.where(x >= 0, 1, 0)
