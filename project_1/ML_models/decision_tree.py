
from sklearn.tree import DecisionTreeClassifier
class DecisionTree():
    def __init__(self, X, Y, max_depth):
        self.model = DecisionTreeClassifier(max_depth=max_depth)
        self.X = X
        self.Y = Y
        #self.max_depth = max_depth

    def train(self):
        self.model.fit(self.X, self.Y)

    def predict(self, X_test):
        return self.model.predict(X_test)