import numpy as np

""" Nearest Neighbor classifier """
class NearestNeighbor:
    def __int__(self):
        pass
    def unpickle(self,file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def train(self, X, y):
        """ Memorize training data """
        self.Xtraining = self.unpickle('train')
        self.ytraining = y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype= self.ytraining.dtype)

        for i in range(num_test):

            distance = np.sum(np.abs(self.Xtraining - X[i,:]), axis=1)
            min_index = np.argmin(distance)
            Ypred[i] = self.ytraining[min_index]

        return Ypred

NearestNeighbor.train([1,2,3,4],[1,4,9,16])