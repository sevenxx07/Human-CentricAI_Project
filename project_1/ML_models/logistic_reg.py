

from sklearn.linear_model import LogisticRegression
class LogisticReg():
    def __init__(self, X, Y, C, max_iter):
        self.model = LogisticRegression(C=C, max_iter=max_iter)
        self.X = X
        self.Y = Y
        #self.C = C
        #self.max_iter = max_iter

    def train(self):
        self.model.fit(self.X, self.Y)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
