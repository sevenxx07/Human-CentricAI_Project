import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.preprocessing import LabelEncoder

df = load_penguins()

class CounterfactualExplainer:
    def __init__(self, model, data, numeric_columns, categorical_columns, mad_values=None, N=500, k=3):
        
        # Model compatibility wrapper (used if the model is missing predict)
        if not hasattr(model, 'predict'):
            class ModelWrapper: 
                def __init__(self, wrapped_model):
                    self.wrapped_model = wrapped_model
                def predict(self, X):
                    print(f"ModelWrapper received input shape: {X.shape}")
                    import numpy as np
                    if isinstance(X, pd.Series):
                        X = X.values.reshape(1, -1)
                    elif isinstance(X, list):
                        X = np.array(X)
                        if X.ndim == 1:
                            X = X.reshape(1, -1)
                    return self.wrapped_model.predict(X)
            self.model = ModelWrapper(model)
        else: 
            self.model = model
        
        
        self.data = data

        if numeric_columns is None: 
            self.numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        else:
            self.numeric_columns = numeric_columns
        
        self.categorical_columns = categorical_columns or []
        self.encoders = {}

        # Fit label encoders for categorical columns
        for cat_col in self.categorical_columns:
            le = LabelEncoder()
            le.fit(data[cat_col])
            self.encoders[cat_col] = le
#_____added
        self.species_encoder = LabelEncoder()
        self.species_encoder.fit(data['species'])
            
        self.N = N
        self.k = k

        # Computing mean absolute deviation 
        if mad_values is not None: 
            self.mad_values = mad_values
        else: 
            self.mad_values = data[self.numeric_columns].apply(lambda x: (x - x.mean()).abs().mean())


    def compute(self, x : pd.Series, target_label):
        neighbors = []

        print("Original input x:")
        print(x)
        print("Target label:", target_label)
        print("MAD values:")
        print(self.mad_values)

        for i in range(self.N):
            x_prime = x.copy()
            print(f"\nIteration {i+1}:")

            print("Perturbing numeric columns:")
            for column in self.numeric_columns:
                noise = np.random.normal(0, 0.5 * max(self.mad_values[column], 1e-3))
                x_prime[column] += noise
                print(f"  {column}: noise={noise:.4f}, new_value={x_prime[column]:.4f}")
            
            print("Perturbing categorical columns:")
            for column in self.categorical_columns:
                current_code = x_prime[column]
                possible_codes = list(set(self.encoders[column].transform(self.data[column].unique())) - {current_code})
                if possible_codes:
                    new_code = np.random.choice(possible_codes)
                    x_prime[column] = new_code
                    print(f"  {column}: changed from {current_code} to {new_code}")
                else:
                    print(f"  {column}: no alternative category found, stays {current_code}")
                
                prediction = self.model.predict(x_prime.values.reshape(1, -1))

                if isinstance(prediction[0], str):
                    prediction = self.species_encoder.transform(prediction)[0]
                else:
                    prediction = prediction[0]

            print(f"Prediction: {prediction}, Target: {target_label}")
    
            if prediction == target_label:
                dist = sum(abs(x[column] - x_prime[column]) / self.mad_values[column] for column in self.numeric_columns)
                changes = {column: round(x_prime[column],2) for column in self.numeric_columns if abs(x[column] - x_prime[column]) > 0.05}
                neighbors.append({"distance": dist, "changes": changes, "original": x.to_dict()})

        # Sort by distance and take top k
        neighbors = sorted(neighbors, key=lambda c: c["distance"])[:self.k]
        return neighbors