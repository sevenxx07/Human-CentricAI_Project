

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

    def evaluate(self, y_test, y_pred):
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }