import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

df = load_penguins()

class CounterfactualExplainer:
    def __init__(self, model, data: pd.DataFrame, numeric_columns, mad_values=None, N=500, k=3):
        self.model = model
        self.data = data

        if numeric_columns is None: 
            self.numeric_columns = data.select_dtypes(include=['number'].columns.tolist())
        else:
            self.numeric_columns = numeric_columns
        
        self.N = N
        self.k = k

        if mad_values is not None: 
            self.mad_values = mad_values
        else: 
            self.mad_values = data[self.numeric_columns].mad()

    def compute(self, x : pd.Series, target_label):
        neighbors = []

        for i in range(self.N):
            x_prime = x.copy()
            
            for column in self.numeric_columns:
                x_prime[column] += np.random.normal(0, 0.2*self.mad_values[column])
            
            prediction = self.model.predict([x_prime.values][0])
            if prediction == target_label:
                dist = sum(abs(x[column] - x_prime[column]) / self.mad_values[column] for column in self.numeric_columns)
                changes = {column: round(x_prime[column],2) for column in self.numeric_columns if abs(x[column] - x_prime[column]) > 0.05}
                neighbors.append({"distance": dist, "changes": changes, "original": x.to_dict()})

        # Sort by distance and take top k
        neighbors = sorted(neighbors, key=lambda c: c["distance"])[:self.k]
        return neighbors