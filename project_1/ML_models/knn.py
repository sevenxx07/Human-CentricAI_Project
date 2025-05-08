

from sklearn.neighbors import KNeighborsClassifier
class KNN():
    def __init__(self, X, Y, k):
        self.model =KNeighborsClassifier(n_neighbors=k)
        self.X = X
        self.Y = Y
        #self.k = k

    def train(self):
        self.model.fit(self.X, self.Y)
    
    def predict(self, X_test):
        return self.model.predict(X_test )