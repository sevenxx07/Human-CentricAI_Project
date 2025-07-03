import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Import your models here
from project_3.Interpretability.Decision_tree_complexity import SparseDecisionTree
from project_3.Interpretability.Decision_tree import PalmerPenguinsDecisionTree
from project_3.Interpretability.Logistic_regression_complexity import SparseLogisticRegression
from project_3.Interpretability.Logistic_regression import PlainLogisticRegressionModel


df = load_penguins()

class CounterfactualExplainer:
    def __init__(self, model, data, numeric_columns, categorical_columns, mad_values=None, N=500, k=3):
        self.model = model
        self.data = data
        self.N = N
        self.k = k
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns

        self.encoders = {}
        for cat_col in self.categorical_columns:
            le = LabelEncoder()
            le.fit(data[cat_col])
            self.encoders[cat_col] = le

        self.species_encoder = LabelEncoder()
        self.species_encoder.fit(data["species"])

        self.mad_values = data[self.numeric_columns].apply(
            lambda x: (x - x.mean()).abs().mean()
        )

    def compute(self, x: pd.Series, target_label):
        neighbors = []

        for i in range(self.N):
            x_prime = x.copy()

            # Perturb numeric columns
            for column in self.numeric_columns:
                if column not in x_prime:
                    continue
                noise = np.random.normal(0, 0.5 * max(self.mad_values[column], 1e-3))
                x_prime[column] += noise

            # Perturb categorical columns
            for column in self.categorical_columns:
                if column not in x_prime:
                    continue
                current_code = x_prime[column]
                possible_codes = list(set(self.encoders[column].transform(self.data[column].unique())) - {current_code})
                if possible_codes:
                    x_prime[column] = np.random.choice(possible_codes)

            # Format for prediction based on model
            try:
                if hasattr(self.model, 'feature_names'):  # You store them in .load_data()
                    input_df = pd.DataFrame([x_prime])
                    if hasattr(self.model, "X_train"):  # Sparse DT or models with .X_train
                        # Ensure column order and add missing columns
                        input_df = input_df.reindex(columns=self.model.feature_names, fill_value=0)
                    else:
                        # For sklearn-based models, do dummy encoding and align
                        input_df = pd.get_dummies(input_df)
                        input_df = input_df.reindex(columns=self.model.feature_names, fill_value=0)

                    pred = self.model.predict(input_df.values)
                else:
                    # fallback
                    pred = self.model.predict(pd.DataFrame([x_prime]))

                pred_label = pred[0] if isinstance(pred, (list, np.ndarray)) else pred

            except Exception as e:
                print(f"Prediction error at iteration {i}: {e}")
                continue

            if pred_label == target_label:
                dist = sum(abs(x[column] - x_prime[column]) / self.mad_values[column]
                           for column in self.numeric_columns if column in x)
                changes = {
                    column: round(x_prime[column], 2)
                    for column in self.numeric_columns
                    if column in x and abs(x[column] - x_prime[column]) > 0.05
                }
                neighbors.append({
                    "distance": dist,
                    "changes": changes,
                    "original": x.to_dict()
                })

        return sorted(neighbors, key=lambda c: c["distance"])[:self.k]

def run_counterfactual_for_model(model_class, model_name, is_sparse=False):
    print(f"\n=== Running Counterfactuals for: {model_name} ===")
    df = load_penguins().dropna()
    numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    categorical_cols = ['island', 'sex']

    # Encode categorical columns
    df_encoded = df.copy()
    for col in categorical_cols:
        df_encoded[col] = pd.Categorical(df_encoded[col]).codes

    # Encode species labels
    species_encoder = LabelEncoder()
    df_encoded["species"] = species_encoder.fit_transform(df_encoded["species"])

    X = df_encoded.drop(columns=['species', 'year'], errors='ignore')
    y = df_encoded['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    # Instantiate model
    if is_sparse:
        alpha = 0.06 if "Tree" in model_name else 30.0
        model = model_class(alpha=alpha)
    else:
        model = model_class()

    # Train the model
    if hasattr(model, "train"):
        model.train()
    else:
        raise RuntimeError(f"Model {model_name} has no train method.")

    # Pick a random sample
    sample_id = np.random.choice(X_test.index)
    x = X_test.loc[sample_id]
    actual = y_test.loc[sample_id]
    target = int((actual + 1) % len(species_encoder.classes_))
    target_name = species_encoder.inverse_transform([target])[0]

    print(f"Sample ID: {sample_id}")
    print(f"Actual: {actual} → Target: {target} ({target_name})")

    # Run counterfactual explanation
    explainer = CounterfactualExplainer(
        model=model,
        data=df_encoded,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        N=500,
        k=3
    )

    results = explainer.compute(x, target)

    if not results:
        print("❌ No counterfactuals found.")
    else:
        for i, r in enumerate(results, 1):
            print(f"\n✅ Counterfactual #{i}")
            print(f"  Distance: {r['distance']}")
            print(f"  Changes: {r['changes']}")

def main():
    run_counterfactual_for_model(PlainLogisticRegressionModel, "Logistic Regression")
    run_counterfactual_for_model(SparseLogisticRegression, "Sparse Logistic Regression", is_sparse=True)
    run_counterfactual_for_model(PalmerPenguinsDecisionTree, "Decision Tree")
    run_counterfactual_for_model(SparseDecisionTree, "Sparse Decision Tree", is_sparse=True)


if __name__ == "__main__":
    main()