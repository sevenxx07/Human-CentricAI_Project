

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

    def evaluate(self, y_test, y_pred):
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }