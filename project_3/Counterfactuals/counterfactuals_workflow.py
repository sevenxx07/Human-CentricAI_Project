import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.preprocessing import LabelEncoder

df = load_penguins()
df = df.dropna()

class CounterfactualExplainer:
    def __init__(self, model, data, numeric_columns, categorical_columns, mad_values=None, N=500, k=3):
        # Model compatibility wrapper 
        if not hasattr(model, 'predict'):
            class ModelWrapper: 
                def __init__(self, wrapped_model):
                    self.wrapped_model = wrapped_model
                def predict(self, X):
                    import numpy as np
                    if isinstance(X, pd.Series):
                        X = X.values.reshape(1, -1)
                    elif isinstance(X, list):
                        X = np.array(X)
                        if X.ndim == 1:
                            X = X.reshape(1, -1)
                    return self.wrapped_model.model.predict(X)
            self.model = ModelWrapper(model)
        else: 
            self.model = model
        
        self.data = data

        if numeric_columns is None: 
            self.numeric_columns = data.select_dtypes(include=['number'].columns.tolist())
        else:
            self.numeric_columns = numeric_columns
        
        self.categorical_columns = categorical_columns or []
        self.encoders = {}

        # Fit label encoders for categorical columns
        for cat_col in self.categorical_columns:
            le = LabelEncoder()
            le.fit(data[cat_col])
            self.encoders[cat_col] = le
            
        self.N = N
        self.k = k

        # Computing mean absolute deviation 
        if mad_values is not None: 
            self.mad_values = mad_values
        else: 
            self.mad_values = data[self.numeric_columns].apply(lambda x: (x - x.mean()).abs().mean())


    def compute(self, x : pd.Series, target_label):
        neighbors = []

        for i in range(self.N):
            x_prime = x.copy()
            
            for column in self.numeric_columns:
                x_prime[column] += np.random.normal(0, 0.2*self.mad_values[column])
            
            prediction = self.model.predict(x_prime.values.reshape(1,-1))
            if prediction == target_label:
                dist = sum(abs(x[column] - x_prime[column]) / self.mad_values[column] for column in self.numeric_columns)
                changes = {column: round(x_prime[column],2) for column in self.numeric_columns if abs(x[column] - x_prime[column]) > 0.05}
                neighbors.append({"distance": dist, "changes": changes, "original": x.to_dict()})

        # Sort by distance and take top k
        neighbors = sorted(neighbors, key=lambda c: c["distance"])[:self.k]
        return neighbors